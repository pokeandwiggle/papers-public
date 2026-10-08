"""Florence-2-Base discrete-token VLA with parallel decoding (BEAST-F style).

Each camera image passes through DaViT and Florence's projector; image tokens
and the tokenized prompt form the BART encoder input. The BART decoder receives
``chunk_size * features`` identical query tokens, uses bidirectional
self-attention, and its tied language-model head scores ``num_bins`` action
tokens per position. Training minimizes per-token cross-entropy; inference
takes the argmax token per position, decodes bin centres to increments and
integrates them from the measured state into absolute 20D TCP actions.
"""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path

import torch
import torch.nn.functional as F
from lerobot.configs import PreTrainedConfig
from lerobot.policies.pretrained import PreTrainedPolicy
from safetensors.torch import load_model
from torch import Tensor

from paw_baseline.florence_vla.actions import (
    FEATURES_PER_ARM,
    ChunkCodec,
    decode_chunk,
    encode_chunk,
    parse_stats,
)
from paw_baseline.florence_vla.configuration_florence_vla import FlorenceVLAConfig
from paw_baseline.layout import CAMERA_KEYS

VLM_DIR = "florence"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
LOC_TOKENS = 1000


def random_shift(images: Tensor, pad: int) -> Tensor:
    """Independent integer shifts of up to ``pad`` pixels (replicate padding)."""
    n, c, h, w = images.shape
    padded = F.pad(images, (pad, pad, pad, pad), mode="replicate")
    oy = torch.randint(0, 2 * pad + 1, (n,), device=images.device)
    ox = torch.randint(0, 2 * pad + 1, (n,), device=images.device)
    rows = oy[:, None] + torch.arange(h, device=images.device)[None]
    cols = ox[:, None] + torch.arange(w, device=images.device)[None]
    return padded[
        torch.arange(n, device=images.device)[:, None, None, None],
        torch.arange(c, device=images.device)[None, :, None, None],
        rows[:, None, :, None],
        cols[:, None, None, :],
    ]


class FlorenceVLAPolicy(PreTrainedPolicy):
    config_class = FlorenceVLAConfig
    name = "florence_vla"

    def __init__(
        self,
        config: FlorenceVLAConfig,
        dataset_stats=None,
        dataset_meta=None,
        *,
        _pretrained_dir: Path | None = None,
        **kwargs,
    ):
        super().__init__(config)
        from transformers import (
            AutoTokenizer,
            Florence2Config,
            Florence2ForConditionalGeneration,
        )

        config.validate_features()
        self.config = config
        if _pretrained_dir is None:
            # Fresh training: public Florence-2 VLM weights, no robotics pretraining.
            self.vlm = Florence2ForConditionalGeneration.from_pretrained(
                config.vlm_repo, revision=config.vlm_revision, dtype=torch.float32
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                config.vlm_repo, revision=config.vlm_revision
            )
        else:
            # Saved checkpoint: architecture and tokenizer travel with the weights.
            vlm_config = Florence2Config.from_pretrained(_pretrained_dir / VLM_DIR)
            self.vlm = Florence2ForConditionalGeneration(vlm_config)
            self.tokenizer = AutoTokenizer.from_pretrained(_pretrained_dir / VLM_DIR)
        self.vlm.float()
        decoder = self.vlm.model.language_model.decoder
        for module in decoder.modules():
            if hasattr(module, "is_causal"):
                module.is_causal = False

        self.codec = ChunkCodec(
            config.action_representation,
            parse_stats(config.action_stats),
            config.chunk_size,
            config.num_bins,
            config.dsd_tau_frac,
            config.dsd_min_direction_weight,
            stats_key=config.stats_key,
        )
        self.features_per_step = 2 * FEATURES_PER_ARM[config.action_representation]
        self.num_positions = config.chunk_size * self.features_per_step
        max_positions = self.vlm.config.text_config.max_position_embeddings
        if self.num_positions > max_positions:
            raise ValueError(
                f"{self.num_positions} action tokens exceed the decoder's {max_positions} positions"
            )
        loc0 = self.tokenizer.convert_tokens_to_ids("<loc_0>")
        if loc0 is None or loc0 == self.tokenizer.unk_token_id:
            raise ValueError("Florence-2 tokenizer lacks <loc_*> tokens")
        # Ordered action bins reuse Florence's ordered location-token embeddings.
        ids = [
            loc0 + round(b * (LOC_TOKENS - 1) / (config.num_bins - 1))
            for b in range(config.num_bins)
        ]
        if len(set(ids)) != config.num_bins:
            raise ValueError("num_bins exceeds the available location tokens")
        self.register_buffer("action_token_ids", torch.tensor(ids), persistent=False)
        self.query_token_id = ids[config.num_bins // 2]
        self.register_buffer(
            "image_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "image_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False
        )
        self.reset()

    # ------------------------------------------------------------------ persistence

    def _save_pretrained(self, save_directory: Path, state_dict=None) -> None:
        super()._save_pretrained(save_directory, state_dict=state_dict)
        self.vlm.config.save_pretrained(Path(save_directory) / VLM_DIR)
        self.tokenizer.save_pretrained(Path(save_directory) / VLM_DIR)

    @classmethod
    def from_pretrained(
        cls, pretrained_name_or_path, *, config=None, revision=None, strict=True, **kwargs
    ):
        directory = Path(pretrained_name_or_path)
        if not directory.is_dir():
            raise ValueError("florence_vla checkpoints load from a local directory")
        if config is None:
            config = PreTrainedConfig.from_pretrained(directory)
        if not isinstance(config, FlorenceVLAConfig):
            raise TypeError("florence_vla loading requires FlorenceVLAConfig")
        kwargs.pop("dataset_stats", None)
        kwargs.pop("dataset_meta", None)
        policy = cls(config, _pretrained_dir=directory)
        missing, unexpected = load_model(
            policy, directory / "model.safetensors", strict=strict, device="cpu"
        )
        if missing or unexpected:
            raise RuntimeError(f"checkpoint mismatch: missing={missing} unexpected={unexpected}")
        policy.to(config.device or "cpu")
        policy.eval()
        return policy

    # ------------------------------------------------------------------ policy API

    def get_optim_params(self) -> dict:
        return self.parameters()

    def reset(self) -> None:
        self._queue: deque[Tensor] = deque(maxlen=self.config.n_action_steps)

    def _prompts(self, batch: dict, size: int) -> list[str]:
        task = batch.get("task")
        if task is None:
            raise ValueError("florence_vla requires a task instruction")
        tasks = [task] * size if isinstance(task, str) else list(task)
        if len(tasks) != size:
            raise ValueError("task batch size mismatch")
        return [self.config.prompt_template.format(task=t.strip()) for t in tasks]

    def _encode(self, batch: dict) -> tuple[Tensor, Tensor]:
        images = torch.stack([batch[key] for key in CAMERA_KEYS], dim=1)
        size, cameras = images.shape[:2]
        pixels = images.flatten(0, 1).float()
        if self.training and self.config.random_shift_pad > 0:
            pixels = random_shift(pixels, self.config.random_shift_pad)
        pixels = F.interpolate(
            pixels,
            size=(self.config.image_size, self.config.image_size),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        pixels = (pixels - self.image_mean) / self.image_std
        image_tokens = self.vlm.model.get_image_features(pixels).pooler_output
        image_tokens = image_tokens.reshape(size, cameras * image_tokens.shape[1], -1)

        text = self.tokenizer(
            self._prompts(batch, size),
            padding=True,
            truncation=True,
            max_length=self.config.max_text_tokens,
            return_tensors="pt",
        ).to(image_tokens.device)
        text_embeds = self.vlm.get_input_embeddings()(text["input_ids"])
        inputs = torch.cat([image_tokens.to(text_embeds.dtype), text_embeds], dim=1)
        mask = torch.cat(
            [
                torch.ones(image_tokens.shape[:2], dtype=text["attention_mask"].dtype, device=inputs.device),
                text["attention_mask"],
            ],
            dim=1,
        )
        encoded = self.vlm.model.language_model.encoder(
            inputs_embeds=inputs, attention_mask=mask
        ).last_hidden_state
        return encoded, mask

    def _action_logits(self, batch: dict) -> Tensor:
        """(B, chunk_size, features_per_step, num_bins) logits."""
        encoded, mask = self._encode(batch)
        size = encoded.shape[0]
        query = torch.full(
            (size, self.num_positions), self.query_token_id, dtype=torch.long, device=encoded.device
        )
        bidirectional = torch.ones(
            (1, 1, self.num_positions, self.num_positions), dtype=torch.bool, device=encoded.device
        ).expand(size, 1, -1, -1)
        hidden = self.vlm.model.language_model.decoder(
            input_ids=query,
            attention_mask=bidirectional,
            encoder_hidden_states=encoded,
            encoder_attention_mask=mask,
            use_cache=False,
        ).last_hidden_state
        weight = self.vlm.lm_head.weight[self.action_token_ids]
        logits = hidden @ weight.to(hidden.dtype).T
        return logits.view(size, self.config.chunk_size, self.features_per_step, -1)

    def forward(self, batch: dict) -> tuple[Tensor, dict]:
        device_type = batch["observation.state"].device.type
        with torch.autocast(device_type=device_type, enabled=False):
            state = batch["observation.state"].double()
            actions = batch["action"].double()
            features = encode_chunk(self.config.action_representation, state, actions)
            targets = self.codec.tokenize(features)
            weights = self.codec.loss_weights(features)
        valid = ~batch["action_is_pad"].bool()
        weights = weights * valid.unsqueeze(-1).to(weights.dtype)

        logits = self._action_logits(batch).float()
        token_loss = F.cross_entropy(
            logits.flatten(0, 2), targets.flatten(), reduction="none"
        ).view_as(targets)
        weights = weights.to(token_loss.dtype)
        loss = (token_loss * weights).sum() / weights.sum().clamp_min(1.0)

        with torch.no_grad():
            predicted = logits.argmax(dim=-1)
            mask = valid.unsqueeze(-1).expand_as(targets)
            accuracy = (predicted == targets)[mask].float().mean()
            within_one = ((predicted - targets).abs() <= 1)[mask].float().mean()
        return loss, {
            "token_accuracy": accuracy.item(),
            "token_within_one": within_one.item(),
            "unweighted_ce": token_loss[mask].mean().item(),
        }

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict, **kwargs) -> Tensor:
        self.eval()
        device_type = batch["observation.state"].device.type
        autocast = device_type == "cuda" and self.config.dtype == "bfloat16"
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=autocast):
            logits = self._action_logits(batch)
        tokens = logits.float().argmax(dim=-1)
        with torch.autocast(device_type=device_type, enabled=False):
            features = self.codec.detokenize(tokens)
            state = batch["observation.state"].double()
            absolute = decode_chunk(
                self.config.action_representation, state, features.to(state.device)
            )
        return absolute.float()

    @torch.no_grad()
    def select_action(self, batch: dict, **kwargs) -> Tensor:
        if not self._queue:
            chunk = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]
            self._queue.extend(chunk.transpose(0, 1))
        return self._queue.popleft()

    def describe(self) -> str:
        return json.dumps(
            {
                "representation": self.config.action_representation,
                "positions": self.num_positions,
                "features_per_step": self.features_per_step,
            }
        )
