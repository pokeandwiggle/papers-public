"""Strict pretrained loading; the PI05 neural network is unchanged."""

import re
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from lerobot.configs import PreTrainedConfig
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from safetensors.torch import load_file

from paw_baseline.pi05.configuration_paw_pi05 import PawPI05Config


def load_strict_weights(model, path: Path) -> None:
    """Use upstream PI key conversion, but never swallow missing/incompatible weights."""
    original = load_file(path)
    fixed = model._fix_pytorch_state_dict_keys(original, model.config)
    weights = {
        key if key.startswith("model.") else "model." + key: value
        for key, value in fixed.items()
    }
    model.load_state_dict(weights, strict=True)


def resolve_weights(source: str, revision: str | None) -> Path:
    """Resolve immutable base weights or an explicit local checkpoint."""
    local = Path(source).is_dir()
    if not local and (
        revision is None or re.fullmatch("[0-9a-f]{40}", revision) is None
    ):
        raise ValueError("PI05 pretrained Hub weights require an immutable commit SHA")
    path = (
        Path(source) / "model.safetensors"
        if local
        else Path(hf_hub_download(source, "model.safetensors", revision=revision))
    )
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


class PawPI05Policy(PI05Policy):
    """Native PI05 network with strict base initialization and checkpoint loading.

    A fresh policy loads its pinned base weights. Thus LeRobot's public
    factory creates our data processors normally; only resume/serve supply a
    pretrained_path and restore the saved checkpoint's processors.
    """

    config_class = PawPI05Config
    name = "paw_pi05"

    def __init__(
        self, config: PawPI05Config, *, _weights_path: Path | None = None, **kwargs
    ):
        path = (
            resolve_weights(config.base_repo, config.base_revision)
            if _weights_path is None
            else _weights_path
        )
        super().__init__(config, **kwargs)
        if config.compile_model:
            # Native PI05 enables TF32 when compiling. Keep the conservative
            # precision default; serving can still explicitly override it after load.
            torch.set_float32_matmul_precision("highest")
        load_strict_weights(self, path)

    @classmethod
    def from_pretrained(
        cls, pretrained_name_or_path, *, config=None, revision=None, **kwargs
    ):
        source = str(pretrained_name_or_path)
        path = resolve_weights(source, revision)
        if config is None:
            config = PreTrainedConfig.from_pretrained(source, revision=revision)
        if not isinstance(config, PawPI05Config):
            raise TypeError("PI05 loading requires the TCP adapter config")
        return cls(config, _weights_path=path, **kwargs)
