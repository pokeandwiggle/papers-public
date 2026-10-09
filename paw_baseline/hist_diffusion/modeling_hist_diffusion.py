"""Diffusion Policy whose U-Net accepts any sequence length.

The FiLM U-Net, encoders, noise schedule and loss are LeRobot's. Two changes
support the strided-history configuration:

* the decoder crops each upsampled feature map to its skip connection and the
  output to the input length, as the authors' reference U-Net does, so the
  past-token sequence ``(n_obs_steps - 1) * obs_stride + prediction_steps``
  need not be a multiple of the downsampling factor;
* emitted actions start after the strided past-token prefix.
"""

import torch
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.diffusion.modeling_diffusion import (
    DiffusionConditionalUnet1d,
    DiffusionModel,
    DiffusionPolicy,
)
from lerobot.utils.constants import OBS_IMAGES, OBS_STATE
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from paw_baseline.hist_diffusion.configuration_hist_diffusion import (
    HistDiffusionConfig,
)


class CroppingConditionalUnet1d(DiffusionConditionalUnet1d):
    """LeRobot's FiLM U-Net with length-preserving crops for odd lengths."""

    def forward(self, x: Tensor, timestep: Tensor | int, global_cond=None) -> Tensor:
        length = x.shape[1]
        x = x.transpose(1, 2)
        timesteps_embed = self.diffusion_step_encoder(timestep)
        if global_cond is not None:
            global_feature = torch.cat([timesteps_embed, global_cond], axis=-1)
        else:
            global_feature = timesteps_embed

        use_gc = self.config.gradient_checkpointing and self.training

        def run(block, value):
            if use_gc:
                return checkpoint(block, value, global_feature, use_reentrant=False)
            return block(value, global_feature)

        encoder_skip_features: list[Tensor] = []
        for resnet, resnet2, downsample in self.down_modules:
            x = run(resnet2, run(resnet, x))
            encoder_skip_features.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = run(mid_module, x)

        for resnet, resnet2, upsample in self.up_modules:
            skip = encoder_skip_features.pop()
            x = torch.cat((x[..., : skip.shape[-1]], skip), dim=1)
            x = run(resnet2, run(resnet, x))
            x = upsample(x)

        x = self.final_conv(x[..., :length])
        return x.transpose(1, 2)


class HistDiffusionModel(DiffusionModel):
    def __init__(self, config: HistDiffusionConfig):
        super().__init__(config)
        unet = CroppingConditionalUnet1d(
            config, global_cond_dim=self._global_cond_dim()
        )
        self.unet = (
            torch.compile(unet, mode=config.compile_mode)
            if config.compile_model
            else unet
        )

    def _global_cond_dim(self) -> int:
        """Per-step state and encoder features times context length, as LeRobot sizes them."""
        config = self.config
        dim = config.robot_state_feature.shape[0]
        if config.image_features:
            encoder = (
                self.rgb_encoder[0]
                if config.use_separate_rgb_encoder_per_camera
                else self.rgb_encoder
            )
            dim += encoder.feature_dim * len(config.image_features)
        if config.env_state_feature:
            dim += config.env_state_feature.shape[0]
        return dim * config.n_obs_steps

    def generate_actions(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        """Return ``n_action_steps`` actions starting at the current observation."""
        batch_size, n_obs_steps = batch[OBS_STATE].shape[:2]
        assert n_obs_steps == self.config.n_obs_steps
        global_cond = self._prepare_global_conditioning(batch)
        actions = self.conditional_sample(batch_size, global_cond=global_cond, noise=noise)
        start = self.config.past_steps
        return actions[:, start : start + self.config.n_action_steps]


class HistDiffusionPolicy(DiffusionPolicy):
    config_class = HistDiffusionConfig
    name = "hist_diffusion"

    def __init__(self, config: HistDiffusionConfig, **kwargs):
        # DiffusionPolicy.__init__ minus its DiffusionModel, which would be discarded.
        PreTrainedPolicy.__init__(self, config)
        config.validate_features()
        self.config = config
        self._queues = None
        self.diffusion = HistDiffusionModel(config)
        self.reset()

    @torch.no_grad()
    def predict_history_chunk(
        self, state: Tensor, images: dict[str, Tensor], noise: Tensor | None = None
    ) -> Tensor:
        """Plan from an explicit, already normalized strided context.

        ``state`` is (B, n_obs_steps, D) and each image (B, n_obs_steps, C, H, W),
        oldest first. Serving owns the history; LeRobot's per-call queues would
        instead hold consecutive calls rather than grid-strided frames.
        """
        batch = {OBS_STATE: state}
        if self.config.image_features:
            batch[OBS_IMAGES] = torch.stack(
                [images[key] for key in self.config.image_features], dim=-4
            )
        return self.diffusion.generate_actions(batch, noise=noise)
