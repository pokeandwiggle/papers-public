"""Serving for attn_diffusion: cached embeddings, delta accumulation, seam interpolation.

The paper's hardware latency controls (A.3): observation embeddings are cached
so each frame passes through each encoder once, and each new chunk is entered
by linear interpolation from the final action of the previous chunk to the
action 180 ms into the new chunk.

``AttnPlanner.plan`` reproduces ``AttnDiffusionPolicy.sample_deltas`` on the
same strided context (tests/test_attn_serving.py checks equality) while
encoding only frames it has not seen, then accumulates the predicted deltas
into absolute targets (``anchor_chunk``).

Anchors. Training anchors the first predicted delta to the measured pose, so
the model also predicts the command's lead over the measured pose. Served
with ``anchor="desired"``, slot 0 is instead the station's current command
(the desired TCP pose of the planning observation) and later slots chain the
predicted per-step deltas, so chunks continue from what the robot is already
tracking. The learned first delta is dropped (a train/serve departure; the
training exports lack the desired pose). Observations without a desired pose
fall back to the measured anchor.
"""

import contextlib
import math

import numpy as np
import torch

from paw_baseline.attn_diffusion.deltas import DELTA_POS, DELTA_ROT, accumulate

# Frame key for the desired TCP pose folded into the 20D model state.
DESIRED_STATE = "anchor.desired_state"
ANCHORS = ("measured", "desired")


def anchor_chunk(frame: dict, future: np.ndarray, anchor: str = "measured") -> tuple[np.ndarray, str]:
    """Absolute (F, 20) targets from (F, 14) current+future deltas; returns the anchor used.

    measured: from the measured pose through the learned first delta (as trained).
    desired: slot 0 holds the desired pose (with the predicted gripper width) and
    slot k adds the predicted per-step deltas 1..k.
    """
    if anchor not in ANCHORS:
        raise ValueError(f"anchor must be one of {ANCHORS}, got {anchor!r}")
    desired = frame.get(DESIRED_STATE) if anchor == "desired" else None
    if desired is None:
        return accumulate(frame["observation.state"], future), "measured"
    future = np.array(future, dtype=np.float64, copy=True)
    for sl in (*DELTA_POS, *DELTA_ROT):
        future[0, sl] = 0.0
    return accumulate(desired, future), "desired"
from paw_baseline.attn_diffusion.modeling_attn_diffusion import STATE_PARTS, camera_name
from paw_baseline.attn_diffusion.reference.timm_obs_encoder import _RANGE_LONG, _RANGE_SHORT

# Wire order: [pose_left(7), pose_right(7), grip_left, grip_right], xyzw quaternions.
_WIRE_QUAT = (slice(3, 7), slice(10, 14))


class AttnPlanner:
    """Episode frame store, embedding cache and planner for one policy.

    Threading: ``append`` and ``snapshot`` run on the observation thread and
    own the frame store; ``sample``/``plan`` run on the inference thread, use
    only the snapshot they are given, and own the embedding cache.
    """

    def __init__(self, policy, anchor: str = "measured"):
        if anchor not in ANCHORS:
            raise ValueError(f"anchor must be one of {ANCHORS}, got {anchor!r}")
        self.policy = policy
        self.config = policy.config
        self.cameras = list(policy.camera_keys)
        self.anchor = anchor
        self.anchors_used = {name: 0 for name in ANCHORS}
        self.reset()

    def reset(self) -> None:
        """Start an episode; call only while no plan is in flight."""
        self.frames: dict[int, dict[str, np.ndarray]] = {}
        self.features: dict[tuple[str, int], dict[str, torch.Tensor]] = {}
        self.count = 0
        self.encoded = 0

    def append(self, frame: dict[str, np.ndarray]) -> int:
        """Store one grid frame ((20,) state, (3, H, W) images) and return its index."""
        index = self.count
        self.frames[index] = frame
        self.count += 1
        oldest = index - self.config.past_steps
        for stale in [i for i in self.frames if 0 < i < oldest]:
            del self.frames[stale]  # frame 0 stays: it pads the context at episode start
        return index

    def context_indices(self, newest: int) -> list[int]:
        """Strided context ending at ``newest``, oldest first; padded with frame 0."""
        offsets = range(-self.config.past_steps, 1, self.config.obs_stride)
        return [max(newest + offset, 0) for offset in offsets]

    def snapshot(self, newest: int) -> dict:
        """The frames one plan needs, captured on the observation thread."""
        indices = self.context_indices(newest)
        return {"newest": newest, "indices": indices, "frames": {i: self.frames[i] for i in set(indices)}}

    def _prune_features(self, newest: int) -> None:
        oldest = newest - self.config.past_steps
        for key in [k for k in self.features if 0 < k[1] < oldest]:
            del self.features[key]

    def _autocast(self, device):
        if self.config.dtype == "bfloat16" and device.type == "cuda":
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    def _rgb_features(
        self, name: str, encoder, indices: list[int], frames: dict, device
    ) -> dict[str, torch.Tensor]:
        """(1, T, D) per camera, encoding each uncached frame once."""
        missing = sorted({i for i in indices if (name, i) not in self.features})
        if missing:
            for key in self.cameras:
                images = torch.from_numpy(np.stack([frames[i][key] for i in missing]))
                images = self.policy.resize_images(images[None].to(device))
                features = encoder._process_rgb(images)[0]  # (n, D), eval center crop
                for row, i in enumerate(missing):
                    self.features.setdefault((name, i), {})[camera_name(key)] = features[row]
            self.encoded += len(missing)
        return {
            camera_name(key): torch.stack([self.features[(name, i)][camera_name(key)] for i in indices])[None]
            for key in self.cameras
        }

    @staticmethod
    def _tokens(encoder, rgb: dict, lowdim: dict, range_index: int):
        """TimmObsEncoder.forward(output_format="dit") from precomputed RGB features."""
        encoded = {key: rgb[key] for key in encoder.rgb_keys}
        for key in encoder.low_dim_keys:
            encoded[key] = encoder.lowdim_projs[key](lowdim[key].float().flatten(2))
        steps = encoder.n_obs_steps
        device = next(iter(encoded.values())).device
        positions = torch.arange(steps, device=device).unsqueeze(0)
        keys = encoder.rgb_keys + encoder.low_dim_keys
        return (
            torch.cat([encoded[k] for k in keys], dim=1),
            torch.cat([positions] * len(keys), dim=1),
            torch.cat(
                [torch.full((1, steps), encoder.key_to_modality[k], dtype=torch.long, device=device) for k in keys],
                dim=1,
            ),
            torch.full((1, steps * len(keys)), range_index, dtype=torch.long, device=device),
        )

    @torch.no_grad()
    def sample(self, snapshot: dict, generator=None) -> torch.Tensor:
        """Unnormalized (1, horizon, 14) deltas for one captured context."""
        policy, config = self.policy, self.config
        core = policy.eval_core
        device = next(core.parameters()).device
        indices, frames = snapshot["indices"], snapshot["frames"]
        self._prune_features(snapshot["newest"])
        short = config.short_range_obs_horizon
        state = torch.from_numpy(np.stack([frames[i]["observation.state"] for i in indices]))
        nstate = policy.normalize_state(state[None].to(device))
        lowdim = {name: nstate[..., sl] for name, sl in STATE_PARTS}
        with self._autocast(device):
            long = self._tokens(
                core.obs_encoder,
                self._rgb_features("long", core.obs_encoder, indices, frames, device),
                lowdim,
                _RANGE_LONG,
            )
            recent = {k: v[:, -short:] for k, v in lowdim.items()}
            short_tokens = self._tokens(
                core.short_range_encoder,
                self._rgb_features("short", core.short_range_encoder, indices[-short:], frames, device),
                recent,
                _RANGE_SHORT,
            )
            cond = tuple(torch.cat([a, b], dim=1) for a, b in zip(long, short_tokens, strict=True))
            ndeltas = policy._conditional_sample(core, cond, generator)
        return policy.unnormalize_deltas(ndeltas.float())

    def plan(self, snapshot: dict, generator=None) -> np.ndarray:
        """Absolute (prediction_steps, 20) model targets from the planning observation."""
        deltas = self.sample(snapshot, generator)[0].cpu().numpy()
        start = self.config.past_steps
        future = deltas[start : start + self.config.prediction_steps]
        absolute, used = anchor_chunk(snapshot["frames"][snapshot["newest"]], future, self.anchor)
        self.anchors_used[used] += 1
        return absolute


def mix_wire(a: np.ndarray, b: np.ndarray, weight: float) -> np.ndarray:
    """Interpolate wire actions: linear positions/grippers, normalized-lerp quaternions."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    out = (1.0 - weight) * a + weight * b
    for sl in _WIRE_QUAT:
        qa = a[sl] if float(np.dot(a[sl], b[sl])) >= 0.0 else -a[sl]
        q = (1.0 - weight) * qa + weight * b[sl]
        out[sl] = q / np.linalg.norm(q)
    return out


def interpolate_into_chunk(
    chunk: np.ndarray,
    seam: int,
    previous: np.ndarray | None,
    interpolation_ms: float,
    slot_ms: float = 1000.0 / 30.0,
) -> np.ndarray:
    """Enter a new (1, T, 16) wire chunk by interpolating from ``previous``.

    ``previous`` is the action the robot executes just before slot ``seam``
    (the first slot the new chunk can still change). Slots from the seam up to
    ``interpolation_ms`` after the chunk's grounding observation are replaced
    by a linear interpolation from ``previous`` to the new chunk's action at
    that time; later slots follow the new chunk. Nothing changes when there is
    no previous action or the seam already lies past that time.
    """
    out = np.array(chunk, dtype=np.float32, copy=True)
    target_slot = interpolation_ms / slot_ms
    if previous is None or seam >= target_slot or out.shape[1] < 2:
        return out
    low = min(int(math.floor(target_slot)), out.shape[1] - 2)
    frac = target_slot - low
    target = mix_wire(out[0, low], out[0, low + 1], frac)
    start = seam - 1  # the previous action sits one slot before the seam
    for slot in range(max(seam, 0), int(math.ceil(target_slot))):
        weight = (slot - start) / (target_slot - start)
        out[0, slot] = mix_wire(previous, target, weight)
    return out
