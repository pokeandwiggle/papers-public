"""Pre-train the frozen action tokenizer of an ``artok`` policy.

Mirrors the reference tokenizer workspaces (train_oattok / train_questtok):
reconstruction MSE on MIN_MAX-normalized action chunks, AdamW (betas 0.9/0.95,
no weight decay), constant learning rate, gradient clipping at 1.0, BF16
autocast and an EMA copy (power 0.75) that becomes the policy's tokenizer.
Chunks start at every selected frame and repeat the episode's last action
past its end, as LeRobot's action queries do during policy training.
"""

import copy
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from safetensors.torch import save_file

TOKENIZER_FILE = "tokenizer.safetensors"
REPORT_PATH = ".paw/tokenizer_report.json"
# LeRobot's NormalizerProcessorStep default epsilon for constant dimensions.
MIN_MAX_EPS = 1e-8


def artok_config_from_args(lerobot_argv: list[str]):
    """Resolve the ARTokConfig that LeRobot will build from the same arguments."""
    import draccus

    from paw_baseline.artok.configuration_artok import ARTokConfig

    policy_args = [
        "--" + arg[len("--policy.") :]
        for arg in lerobot_argv
        if arg.startswith("--policy.") and not arg.startswith("--policy.type=")
    ]
    return draccus.parse(config_class=ARTokConfig, args=policy_args)


def scratch_dir(output_dir: Path, config, prepared_root: Path, episodes: list[int]) -> Path:
    """Per-run directory shared by all ranks on one machine, outside LeRobot's output.

    The key covers the run, tokenizer settings and training data so that a
    concurrent or stale job on the same host cannot supply its tokenizer.
    """
    identity = {
        "output": str(output_dir.resolve()),
        "prepared": str(prepared_root.resolve()),
        "stats": hashlib.sha256((prepared_root / "meta/stats.json").read_bytes()).hexdigest(),
        "episodes": list(episodes),
        "config": {
            key: str(getattr(config, key))
            for key in sorted(vars(config))
            if key.startswith(("tokenizer", "oat_", "fsq", "num_tokens", "horizon"))
        },
    }
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    return Path(tempfile.gettempdir()) / f"paw-artok-{key}"


def normalize_min_max(actions: np.ndarray, stats: dict) -> np.ndarray:
    """LeRobot MIN_MAX normalization: 2 * (x - min) / (max - min) - 1."""
    low = np.asarray(stats["min"], dtype=np.float32)
    high = np.asarray(stats["max"], dtype=np.float32)
    denom = high - low
    denom = np.where(denom == 0, np.float32(MIN_MAX_EPS), denom)
    return (2 * (actions - low) / denom - 1).astype(np.float32)


def load_action_chunks(prepared_root: Path, episodes: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return normalized actions, anchor frame rows and each anchor's episode end row."""
    tables = [
        pq.read_table(path, columns=["action", "episode_index", "frame_index"])
        for path in sorted((prepared_root / "data").glob("chunk-*/file-*.parquet"))
    ]
    actions, episode_index, frame_index = [], [], []
    for table in tables:
        actions.append(
            np.asarray(table["action"].combine_chunks().flatten()).reshape(table.num_rows, -1)
        )
        episode_index.append(table["episode_index"].to_numpy())
        frame_index.append(table["frame_index"].to_numpy())
    actions = np.concatenate(actions).astype(np.float32)
    episode_index = np.concatenate(episode_index)
    frame_index = np.concatenate(frame_index)

    selected = np.isin(episode_index, np.asarray(episodes))
    order = np.lexsort((frame_index[selected], episode_index[selected]))
    actions = actions[selected][order]
    episode_index = episode_index[selected][order]
    if set(np.unique(episode_index).tolist()) != set(episodes):
        raise ValueError("tokenizer episodes are missing from the prepared data")

    stats = json.loads((prepared_root / "meta/stats.json").read_text())["action"]
    actions = normalize_min_max(actions, stats)
    starts = np.flatnonzero(np.r_[True, episode_index[1:] != episode_index[:-1]])
    ends = np.r_[starts[1:], len(episode_index)]
    episode_end = np.repeat(ends, ends - starts)
    return actions, np.arange(len(actions)), episode_end


def _chunks(actions, anchors, ends, rows, horizon):
    offsets = torch.arange(horizon, device=actions.device)
    index = torch.minimum(anchors[rows, None] + offsets, ends[rows, None] - 1)
    return actions[index]


def _ema_decay(step: int, power: float, inv_gamma: float, min_value: float, max_value: float) -> float:
    step = max(0, step - 1)
    if step <= 0:
        return 0.0
    value = 1 - (1 + step / inv_gamma) ** -power
    return max(min_value, min(value, max_value))


@torch.no_grad()
def _evaluate(tokenizer, actions, anchors, ends, horizon, tokenizer_type, num_tokens, generator):
    tokenizer.eval()
    count = min(8192, len(anchors))
    rows = torch.randperm(len(anchors), generator=generator, device="cpu")[:count].to(actions.device)
    target = _chunks(actions, anchors, ends, rows, horizon)
    keeps = [1, 2, 4, 8, num_tokens] if tokenizer_type == "oat" else [num_tokens]
    keeps = sorted({k for k in keeps if k <= num_tokens})
    mse = {}
    for keep in keeps:
        errors = []
        for batch in torch.split(target, 1024):
            errors.append(((tokenizer.autoencode(batch, keep=keep) - batch) ** 2).mean(dim=(1, 2)))
        mse[str(keep)] = float(torch.cat(errors).mean())
    tokens = torch.cat([tokenizer.tokenize(batch) for batch in torch.split(target, 1024)])
    # Detokenizing the full sequence must reproduce the autoencoder path.
    detok = torch.cat([tokenizer.detokenize(t) for t in torch.split(tokens, 1024)])
    # Policy targets are tokenized in train mode (dropout active), as in the
    # reference; record how often they agree with the deterministic tokens.
    tokenizer.train()
    noisy = torch.cat([tokenizer.tokenize(batch) for batch in torch.split(target, 1024)])
    tokenizer.eval()
    return {
        "eval_chunks": count,
        "reconstruction_mse_by_tokens": mse,
        "detokenize_full_mse": float(((detok - target) ** 2).mean()),
        "unique_codes": int(tokens.unique().numel()),
        "unique_codes_per_position": [int(tokens[:, i].unique().numel()) for i in range(tokens.shape[1])],
        "train_mode_token_agreement_per_position": [
            float((noisy[:, i] == tokens[:, i]).float().mean()) for i in range(tokens.shape[1])
        ],
    }


def train_tokenizer(config, prepared_root: Path, episodes: list[int], out_dir: Path, device: str) -> Path:
    """Train the configured tokenizer and write its EMA weights and a report."""
    from paw_baseline.artok.modeling_artok import make_tokenizer

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(config.tokenizer_seed)
    np_actions, np_anchors, np_ends = load_action_chunks(prepared_root, episodes)
    actions = torch.from_numpy(np_actions).to(device)
    anchors = torch.from_numpy(np_anchors).to(device)
    ends = torch.from_numpy(np_ends).to(device)
    action_dim = actions.shape[1]

    tokenizer = make_tokenizer(config, action_dim).to(device)
    ema = copy.deepcopy(tokenizer).requires_grad_(False)
    ema.eval()
    params = [p for p in tokenizer.parameters() if p.requires_grad]
    # Frozen parameters (fixed sin-cos tables) are identical copies already.
    pairs = [
        (p, e)
        for p, e in zip(tokenizer.parameters(), ema.parameters(), strict=True)
        if p.requires_grad
    ]
    trained = [p.data for p, _ in pairs]
    ema_trained = [e.data for _, e in pairs]
    optimizer = torch.optim.AdamW(
        [
            {"params": [p for p in params if p.dim() >= 2], "weight_decay": 0.0},
            {"params": [p for p in params if p.dim() < 2], "weight_decay": 0.0},
        ],
        lr=config.tokenizer_lr,
        betas=tuple(config.optimizer_betas),
    )
    use_bf16 = device.startswith("cuda") and torch.cuda.is_bf16_supported()
    generator = torch.Generator(device="cpu").manual_seed(config.tokenizer_seed)
    batch_size = config.tokenizer_batch_size
    steps_per_epoch = len(anchors) // batch_size
    if steps_per_epoch == 0:
        raise ValueError("tokenizer batch size exceeds the number of training chunks")

    losses, started, permutation = [], time.monotonic(), None
    tokenizer.train()
    for step in range(config.tokenizer_steps):
        position = step % steps_per_epoch
        if position == 0:
            permutation = torch.randperm(len(anchors), generator=generator).to(device)
        rows = permutation[position * batch_size : (position + 1) * batch_size]
        batch = _chunks(actions, anchors, ends, rows, config.horizon)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
            loss = tokenizer.reconstruction_loss(batch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, config.grad_clip_norm)
        optimizer.step()
        decay = _ema_decay(step, config.ema_power, config.ema_inv_gamma, config.ema_min_value, config.ema_max_value)
        with torch.no_grad():
            torch._foreach_mul_(ema_trained, decay)
            torch._foreach_add_(ema_trained, trained, alpha=1 - decay)
        losses.append(loss.item())
        if (step + 1) % 1000 == 0 or step + 1 == config.tokenizer_steps:
            print(
                f"tokenizer[{config.tokenizer_type}] step {step + 1}/{config.tokenizer_steps} "
                f"loss {np.mean(losses[-1000:]):.5f} ({time.monotonic() - started:.0f}s)",
                flush=True,
            )

    report = {
        "tokenizer_type": config.tokenizer_type,
        "steps": config.tokenizer_steps,
        "batch_size": batch_size,
        "lr": config.tokenizer_lr,
        "bf16": use_bf16,
        "train_chunks": len(anchors),
        "episodes": len(episodes),
        "final_train_loss_1k": float(np.mean(losses[-1000:])),
        "seconds": time.monotonic() - started,
        "ema": _evaluate(
            ema, actions, anchors, ends, config.horizon, config.tokenizer_type, config.num_tokens, generator
        ),
    }
    print("tokenizer report: " + json.dumps(report), flush=True)
    (out_dir / "tokenizer_report.json").write_text(json.dumps(report, indent=2) + "\n")
    # Publish the weights last and atomically: other ranks wait for this file.
    path = out_dir / TOKENIZER_FILE
    partial = out_dir / f".{TOKENIZER_FILE}.partial"
    save_file({k: v.detach().contiguous().cpu() for k, v in ema.state_dict().items()}, str(partial))
    os.replace(partial, path)
    return path


def run_tokenizer_stage(distributed_state, config, prepared_root: Path, episodes: list[int], out_dir: Path) -> Path:
    """Train on the main process; other ranks poll for its file.

    Polling, not a collective barrier, spans the stage so NCCL's watchdog
    timeout cannot expire while rank zero trains.
    """
    path = out_dir / TOKENIZER_FILE
    if distributed_state.is_main_process:
        path.unlink(missing_ok=True)
    distributed_state.wait_for_everyone()
    if distributed_state.is_main_process:
        train_tokenizer(config, prepared_root, episodes, out_dir, str(distributed_state.device))
    else:
        while not path.is_file():
            time.sleep(10)
    distributed_state.wait_for_everyone()
    return path


def prefetch_backbone_weights(distributed_state, config) -> None:
    """Download pretrained backbone weights once before every rank loads them."""
    if config.pretrained_backbone_weights is None:
        return
    if distributed_state.is_main_process:
        import torchvision

        torchvision.models.get_weight(config.pretrained_backbone_weights).get_state_dict(
            progress=False
        )
    distributed_state.wait_for_everyone()


def install_tokenizer_hook(lerobot_train, tokenizer_path: Path) -> None:
    """Initialize each freshly created artok policy: tokenizer, then pretrained trunks."""
    original = lerobot_train.make_policy

    def make_policy(*args, **kwargs):
        policy = original(*args, **kwargs)
        policy.load_tokenizer(tokenizer_path)
        if policy.config.pretrained_backbone_weights is not None:
            policy.load_pretrained_backbones()
        return policy

    lerobot_train.make_policy = make_policy
