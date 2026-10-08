"""2D ManiFlow policy: DiT-X trained with flow matching plus consistency training.

Training follows the official reference (geyan21/ManiFlow_Policy@ef2f116,
maniflow/policy/maniflow_image_policy.py and the training workspace) and paper Alg. 2:

* 75% of each batch: flow matching, t ~ 0.999 * Beta(1, 1.5), Δt = 0, target x1 - x0.
* 25%: continuous-time consistency training. t ~ U{0, 1/K, ..., (K-1)/K}, Δt ~ U[0, 1],
  t_next = min(t + Δt, 1). The EMA network predicts the velocity at x_{t_next} for step
  Δt; the extrapolated endpoint x̃1 = x_{t_next} + (1 - t_next) v⁻ gives the student's
  target (x̃1 - x_t) / (1 - t). The step size is passed in relative mode.

Observations are encoded as in the official TransformerObsEncoder path: spatial
ResNet-18 feature-map tokens per camera plus a state token, cross-attended by DiT-X.
The EMA copy of the whole network provides consistency targets and the served weights.
"""

import copy
from collections import deque

import torch
import torch.nn.functional as F
import torchvision
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import get_output_shape, populate_queues
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE
from torch import Tensor, nn

from paw_baseline.ema import EmaSchedule, ema_step
from paw_baseline.maniflow.configuration_maniflow import ManiFlowConfig
from paw_baseline.maniflow.ditx import DiTX


class ManiFlowPolicy(PreTrainedPolicy):
    config_class = ManiFlowConfig
    name = "maniflow"

    def __init__(self, config: ManiFlowConfig, **kwargs):
        super().__init__(config)
        config.validate_features()
        self.config = config
        self._queues = None
        self.model = ManiFlowModel(config)
        # Official workspace: ema_model = copy.deepcopy(model); EMAModel then freezes it.
        self.ema_model = copy.deepcopy(self.model)
        self.ema_model.requires_grad_(False)
        self.register_buffer("ema_optimization_step", torch.zeros((), dtype=torch.long))
        self.reset()

    @classmethod
    def from_pretrained(cls, *args, strict: bool = True, **kwargs):
        """Load strictly: silently missing online or EMA weights would invalidate serving."""
        return super().from_pretrained(*args, strict=strict, **kwargs)

    def get_optim_params(self):
        return self.model.parameters()

    def ema_schedule(self) -> EmaSchedule:
        cfg = self.config
        return EmaSchedule(
            update_after_step=cfg.ema_update_after_step,
            inv_gamma=cfg.ema_inv_gamma,
            power=cfg.ema_power,
            min_value=cfg.ema_min_value,
            max_value=cfg.ema_max_value,
        )

    def update(self) -> None:
        """Called by the LeRobot trainer after every optimizer step."""
        decay = self.ema_schedule().decay(int(self.ema_optimization_step.item()))
        ema_step(self.ema_model, self.model, decay)
        self.ema_optimization_step += 1

    def inference_model(self) -> "ManiFlowModel":
        return self.ema_model if self.config.use_ema_for_inference else self.model

    def reset(self):
        self._queues = {
            OBS_STATE: deque(maxlen=self.config.n_obs_steps),
            OBS_IMAGES: deque(maxlen=self.config.n_obs_steps),
            ACTION: deque(maxlen=self.config.n_action_steps),
        }

    def _stack_images(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        batch = dict(batch)
        batch[OBS_IMAGES] = torch.stack(
            [batch[key] for key in self.config.image_features], dim=-4
        )
        return batch

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        if any(len(queue) > 0 for queue in self._queues.values()):
            batch = {k: torch.stack(list(self._queues[k]), dim=1) for k in batch if k in self._queues}
        else:
            batch = dict(batch)
            for key in self.config.image_features:
                if batch[key].ndim == 4:
                    batch[key] = batch[key].unsqueeze(1)
            if batch[OBS_STATE].ndim == 2:
                batch[OBS_STATE] = batch[OBS_STATE].unsqueeze(1)
            batch = self._stack_images(batch)
        return self.inference_model().generate_actions(batch, noise=noise)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        batch = dict(batch)
        batch.pop(ACTION, None)
        batch = self._stack_images(batch)
        self._queues = populate_queues(self._queues, batch)
        if len(self._queues[ACTION]) == 0:
            actions = self.predict_action_chunk(batch, noise=noise)
            self._queues[ACTION].extend(actions.transpose(0, 1))
        return self._queues[ACTION].popleft()

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        batch = dict(batch)
        for key in self.config.image_features:
            if self.config.n_obs_steps == 1 and batch[key].ndim == 4:
                batch[key] = batch[key].unsqueeze(1)
        if self.config.n_obs_steps == 1 and batch[OBS_STATE].ndim == 2:
            batch[OBS_STATE] = batch[OBS_STATE].unsqueeze(1)
        return self.model.compute_loss(self._stack_images(batch), teacher=self.ema_model.dit)


class ManiFlowObsEncoder(nn.Module):
    """Official TransformerObsEncoder for a ResNet with feature_aggregation=None.

    Each camera (sorted by key, separate backbone per camera) yields its layer-4
    feature map flattened to h*w tokens, projected Linear(C -> n_emb); each
    low-dim input yields one token per observation step, projected
    Linear(D -> n_emb). Tokens are concatenated cameras first, then state. The
    resize/crop preprocessing and backbone construction mirror LeRobot's
    DiffusionRgbEncoder so the Diffusion Policy arm sees identical pixels.
    """

    def __init__(self, config: ManiFlowConfig):
        super().__init__()
        self.config = config
        self.camera_keys = list(config.image_features)
        self.token_order = sorted(range(len(self.camera_keys)), key=lambda i: self.camera_keys[i])

        self.resize = (
            torchvision.transforms.Resize(config.resize_shape) if config.resize_shape is not None else None
        )
        self.do_crop = config.crop_shape is not None
        if self.do_crop:
            self.center_crop = torchvision.transforms.CenterCrop(config.crop_shape)
            self.maybe_random_crop = (
                torchvision.transforms.RandomCrop(config.crop_shape)
                if config.crop_is_random
                else self.center_crop
            )

        if not config.use_separate_rgb_encoder_per_camera:
            raise ValueError("the official encoder uses one backbone per camera (share_rgb_model=False)")
        self.backbones = nn.ModuleList([self._make_backbone(config) for _ in self.camera_keys])

        image_shape = next(iter(config.image_features.values())).shape
        if config.crop_shape is not None:
            dummy_hw = config.crop_shape
        elif config.resize_shape is not None:
            dummy_hw = config.resize_shape
        else:
            dummy_hw = image_shape[1:]
        feature_channels, feature_h, feature_w = get_output_shape(
            self.backbones[0], (1, image_shape[0], *dummy_hw)
        )[1:]
        self.tokens_per_camera = feature_h * feature_w
        self.feature_map_shape = (feature_channels, feature_h, feature_w)
        self.rgb_projections = nn.ModuleList(
            [
                nn.Linear(feature_channels, config.n_emb)
                if feature_channels != config.n_emb
                else nn.Identity()
                for _ in self.camera_keys
            ]
        )
        state_dim = config.robot_state_feature.shape[0]
        self.state_projection = (
            nn.Linear(state_dim, config.n_emb) if state_dim != config.n_emb else nn.Identity()
        )
        self.num_tokens = config.n_obs_steps * (len(self.camera_keys) * self.tokens_per_camera + 1)

    @staticmethod
    def _make_backbone(config: ManiFlowConfig) -> nn.Module:
        model = getattr(torchvision.models, config.vision_backbone)(
            weights=config.pretrained_backbone_weights
        )
        backbone = nn.Sequential(*(list(model.children())[:-2]))
        if config.use_group_norm:
            raise ValueError("GroupNorm replacement is not used with the shared pretrained backbone")
        return backbone

    def _preprocess(self, images: Tensor) -> Tensor:
        if self.resize is not None:
            images = self.resize(images)
        if self.do_crop:
            images = self.maybe_random_crop(images) if self.training else self.center_crop(images)
        return images

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        """batch[OBS_IMAGES]: (B, T, N, C, H, W); batch[OBS_STATE]: (B, T, D) -> (B, L, n_emb)."""
        images = batch[OBS_IMAGES]
        batch_size, n_obs_steps = images.shape[:2]
        tokens = []
        for index in self.token_order:
            camera = images[:, :, index].reshape(batch_size * n_obs_steps, *images.shape[3:])
            feature = self.backbones[index](self._preprocess(camera))
            feature = torch.flatten(feature, start_dim=-2).transpose(1, 2)  # (B*T, h*w, C)
            embedding = self.rgb_projections[index](feature)
            tokens.append(embedding.reshape(batch_size, -1, embedding.shape[-1]))
        state = batch[OBS_STATE].reshape(batch_size, n_obs_steps, -1)
        tokens.append(self.state_projection(state))
        return torch.cat(tokens, dim=1)


class ManiFlowModel(nn.Module):
    def __init__(self, config: ManiFlowConfig):
        super().__init__()
        self.config = config
        self.obs_encoder = ManiFlowObsEncoder(config)
        self.action_dim = config.action_feature.shape[0]
        self.dit = DiTX(
            action_dim=self.action_dim,
            horizon=config.horizon,
            cond_dim=config.n_emb,
            cond_len=self.obs_encoder.num_tokens,
            n_layer=config.n_layer,
            n_head=config.n_head,
            n_emb=config.n_emb,
            mlp_ratio=config.mlp_ratio,
            p_drop_attn=config.p_drop_attn,
            qkv_bias=config.qkv_bias,
            qk_norm=config.qk_norm,
            timestep_embed_dim=config.timestep_embed_dim,
            target_t_embed_dim=config.target_t_embed_dim,
        )

    def _vis_cond(self, batch: dict[str, Tensor]) -> Tensor:
        return self.obs_encoder(batch)

    # ========= time sampling (random draws in the reference order) =========
    def sample_t_flow(self, size: int, device) -> Tensor:
        cfg = self.config
        if cfg.sample_t_mode_flow == "beta":
            dist = torch.distributions.Beta(
                torch.tensor([cfg.beta_alpha], device=device),
                torch.tensor([cfg.beta_beta], device=device),
            )
            return (cfg.beta_cutoff * dist.sample((size, 1, 1))).reshape(size)
        if cfg.sample_t_mode_flow == "lognorm":
            return torch.sigmoid(torch.randn(size, device=device))
        return torch.rand(size, device=device)

    def sample_t_consistency(self, size: int, device) -> Tensor:
        if self.config.sample_t_mode_consistency == "discrete":
            k = self.config.denoise_timesteps
            # The reference draws discrete times with the CPU generator.
            return (torch.randint(low=0, high=k, size=(size,)).float() / k).to(device)
        return torch.rand(size, device=device)

    # ========= training =========
    def flow_targets(self, x1: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Flow matching: instantaneous velocity at t with zero step size."""
        t = self.sample_t_flow(x1.shape[0], x1.device).to(x1.dtype)
        dt = torch.zeros_like(t)
        target_t = dt if self.config.sample_target_t_mode == "relative" else t + dt
        x0 = torch.randn_like(x1)
        x_t = (1 - t[:, None, None]) * x0 + t[:, None, None] * x1
        return x_t, t, target_t, x1 - x0

    @torch.no_grad()
    def consistency_targets(
        self, x1: Tensor, cond: Tensor, teacher: DiTX
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Average-velocity target from the EMA network's endpoint estimate."""
        size = x1.shape[0]
        t = self.sample_t_consistency(size, x1.device).to(x1.dtype)
        delta_t1 = torch.rand((size,), device=x1.device).to(x1.dtype)
        delta_t2 = delta_t1.clone()
        t_next = torch.clamp(t + delta_t1, max=1.0)
        if self.config.sample_target_t_mode == "relative":
            target_t_next, target_t = delta_t2, delta_t1
        else:
            target_t_next, target_t = t_next + delta_t2, t_next
        x0 = torch.randn_like(x1)
        x_t = (1 - t[:, None, None]) * x0 + t[:, None, None] * x1
        x_next = (1 - t_next[:, None, None]) * x0 + t_next[:, None, None] * x1
        v_next = teacher(x_next, t_next, target_t_next, cond)
        pred_x1 = x_next + (1 - t_next[:, None, None]) * v_next
        v_target = (pred_x1 - x_t) / (1 - t[:, None, None])
        return x_t, t, target_t, v_target

    def compute_loss(self, batch: dict[str, Tensor], teacher: DiTX) -> tuple[Tensor, dict]:
        cfg = self.config
        assert batch[OBS_STATE].shape[1] == cfg.n_obs_steps
        actions = batch[ACTION]
        assert actions.shape[1] == cfg.horizon
        vis_cond = self._vis_cond(batch)
        batch_size = actions.shape[0]
        # Equals the reference's int() split for full batches (64 -> 48/16); a
        # short final batch always keeps at least one flow-matching sample.
        ct_bs = int(batch_size * cfg.consistency_batch_ratio)
        flow_bs = batch_size - ct_bs

        x_t, t, target_t, v_target = self.flow_targets(actions[:flow_bs])
        v_pred = self.dit(x_t, t, target_t, vis_cond[:flow_bs])
        loss_flow = F.mse_loss(v_pred, v_target)
        loss = loss_flow
        metrics = {"loss_flow": loss_flow.item()}

        if ct_bs > 0:
            cond = vis_cond[flow_bs : flow_bs + ct_bs]
            x_t, t, target_t, v_target = self.consistency_targets(
                actions[flow_bs : flow_bs + ct_bs], cond.detach(), teacher
            )
            v_pred = self.dit(x_t, t, target_t, cond)
            loss_ct = F.mse_loss(v_pred, v_target)
            loss = loss + loss_ct
            metrics["loss_ct"] = loss_ct.item()
        return loss, metrics

    # ========= inference =========
    @torch.no_grad()
    def generate_actions(
        self,
        batch: dict[str, Tensor],
        noise: Tensor | None = None,
        num_inference_steps: int | None = None,
    ) -> Tensor:
        cfg = self.config
        batch_size, n_obs_steps = batch[OBS_STATE].shape[:2]
        assert n_obs_steps == cfg.n_obs_steps
        vis_cond = self._vis_cond(batch)
        param = next(self.dit.parameters())
        x = (
            noise
            if noise is not None
            else torch.randn(
                (batch_size, cfg.horizon, self.action_dim), dtype=param.dtype, device=param.device
            )
        )
        steps = num_inference_steps or cfg.num_inference_steps
        dt = 1.0 / steps
        for i in range(steps):
            t = torch.full((batch_size,), i / steps, dtype=x.dtype, device=x.device)
            step = torch.full_like(t, dt)
            target_t = step if cfg.sample_target_t_mode == "relative" else t + step
            x = x + self.dit(x, t, target_t, vis_cond) * dt
        start = n_obs_steps - 1
        return x[:, start : start + cfg.n_action_steps]
