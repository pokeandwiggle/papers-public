"""CUPID performance influence for LeRobot Diffusion Policy checkpoints.

Follows Agia et al. (2025), Sec. 5 and App. A, and the official implementation
(agiachris/cupid @2941eba with its TRAK fork):

* Action influence (Eq. 9) between a test pair (s', a') and a training pair
  (s, a) is ``phi(s',a')^T (X^T X)^{-1} phi(s,a)``, where ``phi`` is the randomly
  projected gradient of the label-agnostic output
  ``l_square(s, a) = E_{eps,i} mean ||eps_theta(sqrt(abar_i) a + sqrt(1-abar_i) eps, s, i)||^2``
  over all policy parameters (vision encoders and U-Net). The expectation uses
  B = 64 independent (noise, timestep) draws, ``X`` stacks the projected
  training gradients and ``X^T X`` is TRAK's Gauss-Newton Hessian (lambda = 0,
  one checkpoint, d = 4000).
* Performance influence (Eq. 3, "sum_of_sum" aggregation, "net" metric) of a
  demonstration xi is ``sum_tau R(tau) sum_{(s',a') in tau} sum_{(s,a) in xi}``
  of action influences with R = +1 for successful and -1 for failed rollouts.

Two local adaptations keep the computation within a short local GPU budget:
TRAK's dense JL projection is replaced by an equally unbiased sparse sign
sketch (each coordinate hashed to one of d buckets with a random sign), and each
demonstration's sum over its state-action pairs is estimated from frames on a
fixed stride. Every demonstration uses the same stride, so rankings are not
biased by the subsampling, and per-demonstration sums still scale with length
as in the reference.
"""

import dataclasses
import json
import math
from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np
import torch

from paw_baseline.representation import MODEL_ARM_DIMS, encode_model

NUM_NOISE_DRAWS = 64  # paper A.2: B = 64 (eps, i) samples per pair
PROJ_DIM = 4000  # paper A.2 / reference train_trak.sh: TRAK projection d


@dataclasses.dataclass
class LoadedPolicy:
    policy: object
    preprocess: object
    config: dict
    frozen_arm: str | None
    camera_keys: tuple[str, ...]
    image_shapes: dict[str, tuple[int, int]]
    horizon: int


def load_diffusion_policy(checkpoint: Path, device: str) -> LoadedPolicy:
    """Load a baseline Diffusion checkpoint uncompiled, in eval mode, with its preprocessor."""
    from lerobot.policies import get_policy_class, make_pre_post_processors

    from paw_baseline.serving_metadata import load_serving_metadata
    from paw_baseline.variant import load_policy_config

    config = json.loads((checkpoint / "config.json").read_text())
    if config["type"] != "diffusion":
        raise ValueError(f"CUPID scoring supports diffusion checkpoints, not {config['type']}")
    native = load_policy_config(checkpoint)
    saved_compiled = bool(config.get("compile_model", False))
    native.compile_model = saved_compiled
    policy = get_policy_class("diffusion").from_pretrained(
        checkpoint, config=native, strict=True
    )
    if saved_compiled:
        policy.diffusion.unet = policy.diffusion.unet._orig_mod
    native.compile_model = False
    policy.to(device)
    policy.eval()
    override = {"device": device}
    preprocess, _ = make_pre_post_processors(
        policy.config,
        pretrained_path=checkpoint,
        preprocessor_overrides={"device_processor": override},
        postprocessor_overrides={"device_processor": override},
    )
    cameras = tuple(
        key
        for key, feature in config["input_features"].items()
        if feature["type"] == "VISUAL"
    )
    if tuple(policy.config.image_features) != cameras:
        raise ValueError("camera order differs between config.json and the policy")
    if policy.config.n_obs_steps != 1:
        raise ValueError("CUPID scoring assumes n_obs_steps=1 like baseline serving")
    return LoadedPolicy(
        policy=policy,
        preprocess=preprocess,
        config=config,
        frozen_arm=load_serving_metadata(checkpoint)["frozen_arm"],
        camera_keys=cameras,
        image_shapes={
            key: tuple(config["input_features"][key]["shape"][1:]) for key in cameras
        },
        horizon=int(policy.config.horizon),
    )


class SignSketch:
    """Sparse JL (CountSketch) map R^p -> R^d with one random sign per coordinate.

    Coordinate ``i`` is added with an independent Rademacher sign to bucket
    ``i mod d``. Because the signs are random, ``E[<Sx, Sy>] = <x, y>`` with the
    usual CountSketch variance; the fixed bucket rule keeps the reduction a
    contiguous, coalesced sum. The map is seeded and shared by all samples.
    """

    def __init__(self, num_params: int, proj_dim: int, seed: int, device: str):
        self.num_params = num_params
        self.proj_dim = proj_dim
        self.bucket = math.ceil(num_params / proj_dim)
        self.padded = self.bucket * proj_dim
        generator = torch.Generator(device="cpu").manual_seed(seed)
        self.sign = (
            torch.randint(0, 2, (self.padded,), generator=generator, dtype=torch.int8) * 2 - 1
        ).to(device)

    def __call__(self, flat: torch.Tensor) -> torch.Tensor:
        if flat.shape != (self.num_params,):
            raise ValueError(f"expected ({self.num_params},) gradient, got {tuple(flat.shape)}")
        padded = torch.nn.functional.pad(flat, (0, self.padded - self.num_params))
        return (padded * self.sign).view(self.bucket, self.proj_dim).sum(dim=0)


class ProjectedGradient:
    """phi(s, a): sketched gradient of l_square w.r.t. all policy parameters."""

    def __init__(
        self,
        loaded: LoadedPolicy,
        *,
        noise_seed: int,
        num_draws: int = NUM_NOISE_DRAWS,
        proj_dim: int = PROJ_DIM,
        seed: int = 0,
        device: str = "cuda",
    ):
        """``seed`` fixes the shared projection; ``noise_seed`` the (eps, i) draws.

        Training and test featurization must use different noise seeds: shared
        draws would correlate index-matched training and rollout gradients.
        """
        self.loaded = loaded
        self.diffusion = loaded.policy.diffusion
        self.params = [p for p in self.diffusion.parameters()]
        for param in self.params:
            param.requires_grad_(True)
        self.num_params = sum(p.numel() for p in self.params)
        self.num_draws = num_draws
        self.sketch = SignSketch(self.num_params, proj_dim, seed, device)
        self.generator = torch.Generator(device=device).manual_seed(noise_seed)
        self.device = device

    def output(self, state: torch.Tensor, images: torch.Tensor, action: torch.Tensor):
        """l_square for one normalized pair, averaged over the B noise draws.

        state (D_s,), images (n_cam, C, H, W), action (T, D_a), all normalized.
        """
        diffusion = self.diffusion
        batch = {
            "observation.state": state.reshape(1, 1, -1),
            "observation.images": images.unsqueeze(0).unsqueeze(0),
        }
        global_cond = diffusion._prepare_global_conditioning(batch)
        draws = self.num_draws
        trajectory = action.unsqueeze(0).expand(draws, -1, -1)
        noise = torch.randn(
            trajectory.shape, generator=self.generator, device=trajectory.device
        )
        timesteps = torch.randint(
            0,
            diffusion.noise_scheduler.config.num_train_timesteps,
            (draws,),
            generator=self.generator,
            device=trajectory.device,
        ).long()
        noisy = diffusion.noise_scheduler.add_noise(trajectory, noise, timesteps)
        pred = diffusion.unet(noisy, timesteps, global_cond=global_cond.expand(draws, -1))
        # Reference DiffusionHybridImageFunctionalModelOutput(loss_fn="square"):
        # mean over horizon and action dims of pred^2, averaged over draws.
        return pred.pow(2).mean()

    def __call__(self, state, images, action) -> torch.Tensor:
        loss = self.output(state, images, action)
        grads = torch.autograd.grad(loss, self.params)
        flat = torch.cat([g.reshape(-1) for g in grads])
        return self.sketch(flat)


def normalize_pairs(
    loaded: LoadedPolicy,
    state: torch.Tensor,
    images: dict[str, torch.Tensor],
    action: torch.Tensor,
    task: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply the checkpoint's saved preprocessor to model-space (N, ...) pairs.

    state (N, 20) and action (N, T, 20) are rotation-6D model vectors; images are
    float RGB in [0, 1], (N, 3, H, W). Returns normalized state, stacked images
    (N, n_cam, 3, H, W) and normalized action on the policy device.
    """
    batch = {
        "observation.state": state.float(),
        "action": action.float(),
        "task": [task] * state.shape[0] if isinstance(task, str) else list(task),
        **{key: images[key].float() for key in loaded.camera_keys},
    }
    processed = loaded.preprocess(batch)
    stacked = torch.stack([processed[key] for key in loaded.camera_keys], dim=1)
    return processed["observation.state"], stacked, processed["action"]


def freeze_action_dims(loaded: LoadedPolicy, action: torch.Tensor) -> torch.Tensor:
    """Replace a parked arm's served (live-pinned) actions with its training target.

    Prepared data replaces the parked arm with one constant whose squashed stats
    normalize it to exactly zero; serving pins that arm to the live pose after
    the model. Zero is therefore what the model itself emitted for those dims.
    """
    if loaded.frozen_arm is None:
        return action
    action = action.clone()
    action[..., list(MODEL_ARM_DIMS[loaded.frozen_arm])] = 0.0
    return action


@dataclasses.dataclass
class FeatureSums:
    """Per-group feature sums and (optionally) per-sample features."""

    keys: list
    sums: np.ndarray  # (groups, d) float64
    counts: np.ndarray  # (groups,)
    samples: np.ndarray | None = None  # (N, d) float32 when kept


def featurize_groups(
    featurizer: ProjectedGradient,
    groups: Iterable[tuple[object, Iterator[tuple]]],
    *,
    keep_samples: bool,
    log_every: int = 500,
) -> FeatureSums:
    """Sum phi over each group's normalized pairs (state, images, action)."""
    keys, sums, counts, samples = [], [], [], []
    done = 0
    for key, pairs in groups:
        total = torch.zeros(featurizer.sketch.proj_dim, dtype=torch.float64, device=featurizer.device)
        count = 0
        for state, images, action in pairs:
            phi = featurizer(state, images, action)
            total += phi.double()
            count += 1
            if keep_samples:
                samples.append(phi.float().cpu())
            done += 1
            if log_every and done % log_every == 0:
                print(f"featurized {done} pairs", flush=True)
        keys.append(key)
        sums.append(total.cpu().numpy())
        counts.append(count)
    return FeatureSums(
        keys=keys,
        sums=np.stack(sums) if sums else np.zeros((0, featurizer.sketch.proj_dim)),
        counts=np.asarray(counts),
        samples=torch.stack(samples).numpy() if keep_samples and samples else None,
    )


def gram(train_features: np.ndarray) -> np.ndarray:
    """X^T X of stacked projected training gradients (TRAK's Gauss-Newton matrix)."""
    x = np.asarray(train_features, dtype=np.float64)
    return x.T @ x


def gauss_newton_inverse(xtx: np.ndarray, ridge: float = 0.0) -> np.ndarray:
    """(X^T X + ridge * mean_diag * I)^{-1} in float64 (TRAK uses ridge 0).

    Takes the already accumulated Gram matrix X^T X, never the features.
    """
    xtx = np.asarray(xtx, dtype=np.float64)
    if xtx.ndim != 2 or xtx.shape[0] != xtx.shape[1]:
        raise ValueError(f"expected a square Gram matrix, got {xtx.shape}")
    if not np.allclose(xtx, xtx.T, rtol=1e-6, atol=0):
        raise ValueError("Gram matrix must be symmetric")
    if ridge:
        xtx = xtx + ridge * np.trace(xtx) / xtx.shape[0] * np.eye(xtx.shape[0])
    return np.linalg.inv(xtx)


def performance_influence(
    rollout_sums: np.ndarray,
    rollout_success: np.ndarray,
    demo_sums: np.ndarray,
    xtx_inv: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """CUPID "net" scores: sum_tau R(tau) * trajectory-pair influence.

    Returns (scores (demos,), trajectory influence matrix (rollouts, demos)).
    The pair-sum of phi'^T H^-1 phi over (s',a') in tau and (s,a) in xi equals
    the bilinear form of the two feature sums.
    """
    traj = rollout_sums @ xtx_inv @ demo_sums.T
    signs = np.where(np.asarray(rollout_success, dtype=bool), 1.0, -1.0)
    return signs @ traj, traj


def cupid_scores(
    xtx: np.ndarray,
    demo_sums: np.ndarray,
    rollout_sums: np.ndarray,
    rollout_success: np.ndarray,
    ridge: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Net performance influence from saved X^T X and per-group feature sums."""
    return performance_influence(
        rollout_sums, rollout_success, demo_sums, gauss_newton_inverse(xtx, ridge)
    )


def model_space_rollout(loaded: LoadedPolicy, pairs) -> tuple:
    """Encode physical rollout pairs to model space tensors on CPU."""
    state = torch.from_numpy(encode_model(pairs.state.astype(np.float32)))
    action = torch.from_numpy(encode_model(pairs.action.astype(np.float32)))
    images = {
        key: torch.from_numpy(pairs.images[key]).permute(0, 3, 1, 2).float() / 255.0
        for key in loaded.camera_keys
    }
    return state, images, action


def iter_normalized(loaded: LoadedPolicy, state, images, action, task: str, chunk: int = 64):
    """Yield normalized single pairs in chunks through the saved preprocessor."""
    for start in range(0, state.shape[0], chunk):
        end = start + chunk
        s, im, a = normalize_pairs(
            loaded,
            state[start:end],
            {key: value[start:end] for key, value in images.items()},
            action[start:end],
            task,
        )
        a = freeze_action_dims(loaded, a)
        for index in range(s.shape[0]):
            yield s[index], im[index], a[index]
