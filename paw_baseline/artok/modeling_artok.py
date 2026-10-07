"""Autoregressive policy over discrete action tokens (reference OATPolicy/QueSTPolicy).

The frozen tokenizer maps normalized action chunks to ``num_tokens`` tokens.
A causal transformer, conditioned on robomimic-style image and state features,
predicts the tokens with teacher forcing; inference samples them with top-k
and decodes the chunk. OAT and QueST policies differ only in the tokenizer.
"""

import copy
from collections import deque
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision
from lerobot.policies.diffusion.modeling_diffusion import SpatialSoftmax
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import get_output_shape
from lerobot.utils.constants import ACTION, OBS_STATE
from safetensors.torch import load_file
from torch import Tensor, nn

from paw_baseline.artok.ar_model import AutoregressiveModel
from paw_baseline.artok.configuration_artok import ARTokConfig
from paw_baseline.artok.oat_tokenizer import OATTokenizer
from paw_baseline.artok.quest_tokenizer import QueSTTokenizer


def make_tokenizer(config: ARTokConfig, action_dim: int) -> nn.Module:
    """Build the configured tokenizer for normalized (B, horizon, action_dim) chunks."""
    if config.tokenizer_type == "oat":
        return OATTokenizer(
            action_dim=action_dim,
            horizon=config.horizon,
            emb_dim=config.tokenizer_dim,
            head_dim=config.tokenizer_head_dim,
            encoder_depth=config.tokenizer_encoder_depth,
            decoder_depth=config.tokenizer_decoder_depth,
            pdropout=config.tokenizer_dropout,
            num_registers=config.num_tokens,
            fsq_levels=tuple(config.fsq_levels),
            token_dropout_mode=config.oat_token_dropout_mode,
            use_causal_decoder=config.tokenizer_causal_decoder,
        )
    return QueSTTokenizer(
        action_dim=action_dim,
        horizon=config.horizon,
        encoder_dim=config.tokenizer_dim,
        encoder_nhead=config.tokenizer_dim // config.tokenizer_head_dim,
        encoder_nlayer=config.tokenizer_encoder_depth,
        decoder_dim=config.tokenizer_dim,
        decoder_nhead=config.tokenizer_dim // config.tokenizer_head_dim,
        decoder_nlayer=config.tokenizer_decoder_depth,
        use_causal_decoder=config.tokenizer_causal_decoder,
        downsample_factor=config.horizon // config.num_tokens,
        dropout=config.tokenizer_dropout,
        fsq_levels=tuple(config.fsq_levels),
    )


def _replace_batchnorm(module: nn.Module) -> nn.Module:
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            setattr(
                module,
                name,
                nn.GroupNorm(num_groups=child.num_features // 16, num_channels=child.num_features),
            )
        else:
            _replace_batchnorm(child)
    return module


def crop_images(images: Tensor, shape: tuple[int, int], random: bool) -> Tensor:
    """Per-sample random crops (robomimic CropRandomizer) or a center crop."""
    batch, channels, height, width = images.shape
    crop_h, crop_w = shape
    if random:
        top = torch.randint(0, height - crop_h + 1, (batch,), device=images.device)
        left = torch.randint(0, width - crop_w + 1, (batch,), device=images.device)
    else:
        top = torch.full((batch,), (height - crop_h) // 2, device=images.device)
        left = torch.full((batch,), (width - crop_w) // 2, device=images.device)
    rows = top[:, None] + torch.arange(crop_h, device=images.device)
    cols = left[:, None] + torch.arange(crop_w, device=images.device)
    return images[
        torch.arange(batch, device=images.device)[:, None, None, None],
        torch.arange(channels, device=images.device)[None, :, None, None],
        rows[:, None, :, None],
        cols[:, None, None, :],
    ]


def resnet_trunk(name: str, weights: str | None = None) -> nn.Sequential:
    """torchvision ResNet without its pooling and classifier layers."""
    backbone = getattr(torchvision.models, name)(weights=weights)
    return nn.Sequential(*list(backbone.children())[:-2])


class RgbEncoder(nn.Module):
    """robomimic VisualCore: ResNet trunk, spatial softmax, linear.

    The trunk is built randomly initialized; pretrained weights are loaded by
    ``ARTokPolicy.load_pretrained_backbones`` at the start of training.
    """

    def __init__(self, config: ARTokConfig, image_shape: tuple[int, int, int]):
        super().__init__()
        self.resize = (
            torchvision.transforms.Resize(tuple(config.resize_shape), antialias=True)
            if config.resize_shape is not None
            else None
        )
        self.crop_shape = tuple(config.crop_shape) if config.crop_shape is not None else None
        self.crop_is_random = config.crop_is_random
        self.backbone = resnet_trunk(config.vision_backbone)
        if config.use_group_norm:
            self.backbone = _replace_batchnorm(self.backbone)
        hw = (
            self.crop_shape
            or (tuple(config.resize_shape) if config.resize_shape else None)
            or tuple(image_shape[1:])
        )
        # Probe in eval mode so BatchNorm neither needs batch statistics nor
        # updates its running statistics with the dummy input.
        self.backbone.eval()
        with torch.no_grad():
            feature_map = get_output_shape(self.backbone, (1, image_shape[0], *hw))[1:]
        self.backbone.train()
        self.pool = SpatialSoftmax(feature_map, num_kp=config.spatial_softmax_num_keypoints)
        self.out = nn.Linear(config.spatial_softmax_num_keypoints * 2, config.image_feature_dim)

    def forward(self, images: Tensor) -> Tensor:
        if self.resize is not None:
            images = self.resize(images)
        if self.crop_shape is not None:
            images = crop_images(images, self.crop_shape, self.training and self.crop_is_random)
        return self.out(torch.flatten(self.pool(self.backbone(images)), start_dim=1))


class ObservationEncoder(nn.Module):
    """Concatenate per-camera image features (separate encoders) and the state."""

    def __init__(self, config: ARTokConfig):
        super().__init__()
        self.camera_keys = list(config.image_features)
        image_shape = tuple(next(iter(config.image_features.values())).shape)
        self.cameras = nn.ModuleList([RgbEncoder(config, image_shape) for _ in self.camera_keys])
        self.feature_dim = (
            len(self.camera_keys) * config.image_feature_dim + config.robot_state_feature.shape[0]
        )

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        features = [
            encoder(batch[key]) for key, encoder in zip(self.camera_keys, self.cameras, strict=True)
        ]
        features.append(batch[OBS_STATE])
        return torch.cat(features, dim=-1).unsqueeze(1)  # (B, n_obs_steps=1, F)


class ARTokPolicy(PreTrainedPolicy):
    config_class = ARTokConfig
    name = "artok"

    def __init__(self, config: ARTokConfig, **kwargs):
        super().__init__(config)
        config.validate_features()
        self.config = config
        action_dim = config.action_feature.shape[0]

        self.tokenizer = make_tokenizer(config, action_dim)
        self.tokenizer.requires_grad_(False)
        self.codebook_size = self.tokenizer.codebook_size
        self.bos_id = self.codebook_size

        self.obs_encoder = ObservationEncoder(config)
        self.model = AutoregressiveModel(
            vocab_size=self.codebook_size + 1,
            max_seq_len=config.num_tokens + 1,
            max_cond_len=config.n_obs_steps,
            cond_dim=self.obs_encoder.feature_dim,
            n_layer=config.n_layers,
            n_head=config.n_heads,
            n_emb=config.embed_dim,
            p_drop_emb=config.dropout,
            p_drop_attn=config.dropout,
        )
        if config.use_ema:
            self.ema_obs_encoder = copy.deepcopy(self.obs_encoder).requires_grad_(False)
            self.ema_model = copy.deepcopy(self.model).requires_grad_(False)
            self.register_buffer("ema_step", torch.zeros((), dtype=torch.long))
        self.reset()

    @classmethod
    def _load_as_safetensor(cls, model, model_file, map_location, strict):
        """Always load strictly: tokenizer and EMA weights must be present."""
        return super()._load_as_safetensor(model, model_file, map_location, True)

    def load_tokenizer(self, path: Path) -> None:
        """Install pre-trained tokenizer weights (training-time only)."""
        self.tokenizer.load_state_dict(load_file(str(path)), strict=True)

    def load_pretrained_backbones(self) -> None:
        """Load torchvision weights into every camera trunk, online and EMA (training-time only)."""
        weights = self.config.pretrained_backbone_weights
        if weights is None:
            raise ValueError("no pretrained_backbone_weights configured")
        state = resnet_trunk(self.config.vision_backbone, weights).state_dict()
        encoders = [self.obs_encoder]
        if self.config.use_ema:
            encoders.append(self.ema_obs_encoder)
        for encoder in encoders:
            for camera in encoder.cameras:
                camera.backbone.load_state_dict(state, strict=True)

    def get_optim_params(self) -> list[dict]:
        return [
            {"params": [p for p in self.model.parameters() if p.requires_grad]},
            {
                "params": [p for p in self.obs_encoder.parameters() if p.requires_grad],
                "lr": self.config.obs_encoder_lr,
            },
        ]

    def reset(self):
        self._action_queue = deque(maxlen=self.config.n_action_steps)

    def _ema_decay(self, step: int) -> float:
        """Reference EMAModel.get_decay with update_after_step=0."""
        step = max(0, step - 1)
        if step <= 0:
            return 0.0
        value = 1 - (1 + step / self.config.ema_inv_gamma) ** -self.config.ema_power
        return max(self.config.ema_min_value, min(value, self.config.ema_max_value))

    @torch.no_grad()
    def update(self) -> None:
        """Called by LeRobot after every optimizer step."""
        if not self.config.use_ema:
            return
        decay = self._ema_decay(int(self.ema_step.item()))
        seen = set()  # tied weights (token embedding / head) update once
        for online, averaged in ((self.obs_encoder, self.ema_obs_encoder), (self.model, self.ema_model)):
            for module, ema_module in zip(online.modules(), averaged.modules(), strict=True):
                for param, ema_param in zip(
                    module.parameters(recurse=False), ema_module.parameters(recurse=False), strict=True
                ):
                    if id(param) in seen:
                        continue
                    seen.add(id(param))
                    # Reference EMAModel: BatchNorm and frozen parameters are copied.
                    if isinstance(module, nn.modules.batchnorm._BatchNorm) or not param.requires_grad:
                        ema_param.copy_(param.data)
                    else:
                        ema_param.mul_(decay).add_(param.data, alpha=1 - decay)
                # BatchNorm running statistics must follow the online model,
                # otherwise the EMA encoder normalizes with stale statistics.
                for buffer, ema_buffer in zip(
                    module.buffers(recurse=False), ema_module.buffers(recurse=False), strict=True
                ):
                    ema_buffer.copy_(buffer)
        self.ema_step += 1

    def _inference_modules(self) -> tuple[nn.Module, nn.Module]:
        if self.config.use_ema:
            return self.ema_obs_encoder, self.ema_model
        return self.obs_encoder, self.model

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        """Sample tokens and decode a normalized (B, horizon, action_dim) chunk."""
        obs_encoder, model = self._inference_modules()
        cond = obs_encoder(batch)
        prefix = torch.full((cond.shape[0], 1), self.bos_id, dtype=torch.long, device=cond.device)
        num_tokens = self.config.decode_tokens or self.config.num_tokens
        tokens = model.generate(
            prefix,
            cond=cond,
            max_new_tokens=num_tokens,
            temperature=self.config.temperature,
            top_k=self.config.top_k,
        )[:, 1:]
        return self.tokenizer.detokenize(tokens)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        if len(self._action_queue) == 0:
            chunk = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]
            self._action_queue.extend(chunk.transpose(0, 1))
        return self._action_queue.popleft()

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """Next-token cross-entropy on the frozen tokenizer's action tokens.

        The tokenizer follows the policy's train/eval mode, as in the reference
        (its dropout is active while tokenizing training targets).
        """
        with torch.no_grad():
            tokens = self.tokenizer.tokenize(batch[ACTION])
        cond = self.obs_encoder(batch)
        bos = torch.full((tokens.shape[0], 1), self.bos_id, dtype=torch.long, device=tokens.device)
        logits = self.model(torch.cat([bos, tokens[:, :-1]], dim=1), cond)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), tokens.reshape(-1))
        accuracy = (logits.argmax(-1) == tokens).float().mean()
        return loss, {"token_accuracy": accuracy.item()}
