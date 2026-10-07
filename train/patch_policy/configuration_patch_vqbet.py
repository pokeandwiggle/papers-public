"""Configure Patch Policy (VQ-BeT head) over frozen DINOv2 ViT-S/14 features.

``visual_feature="patch"`` feeds all 16x16 patch tokens of every camera and
history frame through a block-causal transformer (the method);
``visual_feature="cls"`` feeds one CLS token per camera and frame through the
identical policy (the paper's global-feature control). Observations are images
only: proprioceptive state is carried by the dataset contract but unused.

Time runs on the 30 Hz action grid. History frames are ``obs_stride`` steps
apart, each frame predicts a ``chunk_size``-step action chunk, and serving
replans every ``replan_every`` observations from the newest frame.
"""

from dataclasses import dataclass, field

from lerobot.configs import NormalizationMode, PreTrainedConfig
from lerobot.optim import AdamWConfig

from paw_baseline.layout import CAMERA_KEYS
from paw_baseline.representation import MODEL_DIM

DINOV2_REPO = "facebook/dinov2-small"
DINOV2_REVISION = "ed25f3a31f01632728cabb09d1542f84ab7b0056"
VISUAL_FEATURES = ("patch", "cls")


@PreTrainedConfig.register_subclass("patch_vqbet")
@dataclass
class PatchVQBeTConfig(PreTrainedConfig):
    # Observation history (paper real-robot "window size" 2).
    n_obs_steps: int = 2
    obs_stride: int = 3
    # Prediction horizon: VQ-BeT action window, in 30 Hz steps.
    chunk_size: int = 30
    # Actions emitted per plan; serving emits the whole predicted chunk.
    n_action_steps: int = 30
    # Serving replans after this many 30 Hz observations.
    replan_every: int = 3

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.IDENTITY,
            "ACTION": NormalizationMode.MIN_MAX,
        }
    )

    # Frozen visual encoder.
    visual_feature: str = "patch"
    encoder_repo: str = DINOV2_REPO
    encoder_revision: str = DINOV2_REVISION
    # False only for offline tests: a fresh policy then keeps random encoder weights.
    encoder_pretrained: bool = True
    image_size: int = 224
    # Autocast dtype for the frozen encoder and transformer trunk; the VQ
    # decoder, heads and losses always run in float32.
    trunk_dtype: str = "bfloat16"

    # Block-causal GPT trunk (paper Table 10 real robot: N=8, heads=8, d=384).
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 384
    gpt_output_dim: int = 256
    dropout: float = 0.0

    # Residual VQ-VAE tokenizer.
    vqvae_latent_dim: int = 512
    vqvae_n_embed: int = 16
    vqvae_groups: int = 2
    vqvae_hidden_dim: int = 128
    vqvae_hidden_layers: int = 1
    vqvae_steps: int = 20000
    vqvae_batch_size: int = 2048
    vqvae_lr: float = 1e-3
    vqvae_weight_decay: float = 1e-4

    # VQ-BeT losses.
    primary_code_loss_weight: float = 5.0
    secondary_code_loss_weight: float = 0.5
    offset_loss_weight: float = 100.0
    focal_gamma: float = 2.0

    # Optimizer (paper Table 10: lr 3e-4, weight decay 1e-4); constant LR and
    # no gradient clipping, as in the reference trainer.
    optimizer_lr: float = 3e-4
    optimizer_betas: tuple[float, float] = (0.9, 0.999)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-4
    # The reference adds the code-bin head to its GPT AdamW without an explicit
    # decay, so it inherits torch's AdamW default of 1e-2.
    bin_head_weight_decay: float = 1e-2
    optimizer_grad_clip_norm: float = 0.0

    drop_n_last_frames: int = 0

    def __post_init__(self):
        super().__post_init__()
        if self.visual_feature not in VISUAL_FEATURES:
            raise ValueError(f"visual_feature must be one of {VISUAL_FEATURES}")
        if self.n_obs_steps < 1 or self.obs_stride < 1 or self.replan_every < 1:
            raise ValueError("n_obs_steps, obs_stride and replan_every must be positive")
        if not 1 <= self.n_action_steps <= self.chunk_size:
            raise ValueError("n_action_steps must be within [1, chunk_size]")
        if self.vqvae_groups < 2:
            raise ValueError("VQ-BeT uses a primary and at least one secondary code")
        if self.image_size % 14:
            raise ValueError("image_size must be a multiple of the ViT-S/14 patch size")
        if self.trunk_dtype not in ("bfloat16", "float32"):
            raise ValueError("trunk_dtype must be bfloat16 or float32")

    @property
    def history_offsets(self) -> list[int]:
        """30 Hz offsets of the history frames relative to the newest frame."""
        return [-(self.n_obs_steps - 1 - i) * self.obs_stride for i in range(self.n_obs_steps)]

    @property
    def tokens_per_camera(self) -> int:
        side = self.image_size // 14
        return side * side if self.visual_feature == "patch" else 1

    @property
    def observation_delta_indices(self) -> list[int]:
        return self.history_offsets

    @property
    def action_delta_indices(self) -> list[int]:
        # Every history frame is supervised with the chunk starting at its own time.
        return list(range(self.history_offsets[0], self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self) -> None:
        return None

    def validate_features(self) -> None:
        if tuple(self.image_features) != CAMERA_KEYS:
            raise ValueError(f"Patch Policy requires cameras {CAMERA_KEYS} in order")
        shapes = {tuple(ft.shape) for ft in self.image_features.values()}
        if len(shapes) != 1:
            raise ValueError("all cameras must share one image shape")
        if self.action_feature is None or tuple(self.action_feature.shape) != (MODEL_DIM,):
            raise ValueError(f"Patch Policy requires {MODEL_DIM}D TCP actions")
