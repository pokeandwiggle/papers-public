"""Configure an autoregressive action-token policy with an OAT or QueST tokenizer.

Defaults follow the OAT paper and reference configs (train_oatpolicy.yaml,
train_questpolicy.yaml, train_oattok.yaml, train_questtok.yaml): H_a=32,
H_l=8, FSQ [8,5,5,5], 4-layer/256-d AR decoder, top-k 10 sampling, AdamW
(betas 0.9/0.95, no weight decay), constant LR, lr 5e-5 (policy and
tokenizer) and 1e-5 (observation encoder), gradient clip 1.0, EMA power 0.75.
"""

from dataclasses import dataclass, field

from lerobot.configs import NormalizationMode, PreTrainedConfig
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import ConstantWithWarmupSchedulerConfig

TOKENIZER_TYPES = ("oat", "quest")


@PreTrainedConfig.register_subclass("artok")
@dataclass
class ARTokConfig(PreTrainedConfig):
    n_obs_steps: int = 1
    horizon: int = 32
    n_action_steps: int = 16
    drop_n_last_frames: int = 0

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MIN_MAX,
            "STATE": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
        }
    )

    # Action tokenizer. Both tokenizers emit `num_tokens` tokens over a
    # codebook of prod(fsq_levels) entries.
    tokenizer_type: str = "oat"
    num_tokens: int = 8
    fsq_levels: tuple[int, ...] = (8, 5, 5, 5)
    tokenizer_dim: int = 256
    tokenizer_head_dim: int = 64
    tokenizer_encoder_depth: int = 2
    tokenizer_decoder_depth: int = 4
    tokenizer_dropout: float = 0.1
    oat_token_dropout_mode: str = "pow2"
    tokenizer_causal_decoder: bool = True

    # Tokenizer pre-training (run by train/ before policy training).
    tokenizer_steps: int = 40000
    tokenizer_batch_size: int = 256
    tokenizer_lr: float = 5e-5
    tokenizer_seed: int = 42

    # Autoregressive policy.
    embed_dim: int = 256
    n_layers: int = 4
    n_heads: int = 4
    dropout: float = 0.1
    temperature: float = 1.0
    top_k: int = 10
    # Tokens generated at inference; None decodes the full sequence.
    decode_tokens: int | None = None

    # Observation encoder (robomimic ResNet18 + spatial softmax). The reference
    # trains it from scratch with GroupNorm. Pretrained torchvision weights
    # (e.g. "ResNet18_Weights.IMAGENET1K_V1") keep their BatchNorm layers and
    # use ImageNet mean/std input normalization; train/ loads them into fresh
    # policies, so serving never downloads them.
    vision_backbone: str = "resnet18"
    pretrained_backbone_weights: str | None = None
    use_group_norm: bool = True
    resize_shape: tuple[int, int] | None = (128, 128)
    crop_shape: tuple[int, int] | None = (116, 116)
    crop_is_random: bool = True
    spatial_softmax_num_keypoints: int = 32
    image_feature_dim: int = 64

    # Optimization.
    optimizer_lr: float = 5e-5
    obs_encoder_lr: float = 1e-5
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_weight_decay: float = 0.0
    grad_clip_norm: float = 1.0
    # The reference "constant" diffusers schedule ignores lr_warmup_steps.
    scheduler_warmup_steps: int = 0
    # Mixed precision used by LeRobot's Accelerator (reference: allow_bf16).
    dtype: str = "bfloat16"

    # EMA of the trained policy weights, used for inference.
    use_ema: bool = True
    ema_inv_gamma: float = 1.0
    ema_power: float = 0.75
    ema_min_value: float = 0.0
    ema_max_value: float = 0.9999

    def __post_init__(self):
        super().__post_init__()
        if self.tokenizer_type not in TOKENIZER_TYPES:
            raise ValueError(f"tokenizer_type must be one of {TOKENIZER_TYPES}")
        if self.n_obs_steps != 1:
            raise ValueError("baseline serving supports n_obs_steps=1")
        if not 1 <= self.n_action_steps <= self.horizon:
            raise ValueError("n_action_steps must be in [1, horizon]")
        if self.tokenizer_type == "quest" and self.horizon // 4 != self.num_tokens:
            raise ValueError("QueST downsamples the horizon by 4 into num_tokens")
        if self.decode_tokens is not None:
            if not 1 <= self.decode_tokens <= self.num_tokens:
                raise ValueError("decode_tokens must be in [1, num_tokens]")
            if self.tokenizer_type == "quest" and self.decode_tokens != self.num_tokens:
                raise ValueError("QueST cannot decode a token prefix")
        if not self.vision_backbone.startswith("resnet"):
            raise ValueError("vision_backbone must be a torchvision ResNet")
        if self.pretrained_backbone_weights is not None:
            if self.use_group_norm:
                raise ValueError(
                    "pretrained backbone weights require BatchNorm (use_group_norm=false)"
                )
            # Pretrained ResNets expect ImageNet-standardized inputs; the
            # training wrapper supplies ImageNet mean/std as the visual stats.
            self.normalization_mapping = {
                **self.normalization_mapping,
                "VISUAL": NormalizationMode.MEAN_STD,
            }

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.grad_clip_norm,
        )

    def get_scheduler_preset(self) -> ConstantWithWarmupSchedulerConfig:
        return ConstantWithWarmupSchedulerConfig(num_warmup_steps=self.scheduler_warmup_steps)

    def validate_features(self) -> None:
        if not self.image_features:
            raise ValueError("artok requires at least one camera")
        shapes = {tuple(ft.shape) for ft in self.image_features.values()}
        if len(shapes) != 1:
            raise ValueError("all cameras must share one image shape")
        if self.robot_state_feature is None or self.action_feature is None:
            raise ValueError("artok requires observation.state and action")

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.horizon))

    @property
    def reward_delta_indices(self) -> None:
        return None
