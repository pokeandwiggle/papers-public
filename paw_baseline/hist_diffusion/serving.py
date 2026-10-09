"""Serving-side history and replanning for ``hist_diffusion`` checkpoints.

The robot streams a dense, hold-last 30 Hz observation grid. The server keeps
the newest frames, assembles the strided context the policy was trained on,
and plans from a new observation every ``replan_steps`` grid steps. Each plan
emits the full prediction grounded at its observation; the next plan
supersedes it, so ``replan_steps`` actions run open loop per plan.
"""

from collections import deque

import numpy as np

# Wire order: [pose_left(7), pose_right(7), grip_left, grip_right], xyzw quaternions.
# Positions and grippers interpolate linearly; quaternions use normalized lerp.
_WIRE_QUAT = (slice(3, 7), slice(10, 14))


class StridedHistory:
    """Newest frames of one episode and their strided context.

    Frames before the first observation repeat it, matching LeRobot's clamping
    of context indices at episode starts during training.
    """

    def __init__(self, n_obs_steps: int, obs_stride: int):
        if n_obs_steps <= 0 or obs_stride <= 0:
            raise ValueError("n_obs_steps and obs_stride must be positive")
        self.n_obs_steps = n_obs_steps
        self.obs_stride = obs_stride
        self._span = (n_obs_steps - 1) * obs_stride
        self.reset()

    def reset(self) -> None:
        self._frames: deque[dict[str, np.ndarray]] = deque(maxlen=self._span + 1)
        self._first: dict[str, np.ndarray] | None = None
        self.count = 0

    def append(self, frame: dict[str, np.ndarray]) -> int:
        """Store one grid frame and return its index within the episode."""
        if self._first is None:
            self._first = frame
        self._frames.append(frame)
        self.count += 1
        return self.count - 1

    def context(self) -> dict[str, np.ndarray]:
        """Stack the strided context ending at the newest frame, oldest first."""
        if self._first is None:
            raise RuntimeError("history is empty")
        newest = self.count - 1
        oldest_held = newest - len(self._frames) + 1
        frames = []
        for offset in range(-self._span, 1, self.obs_stride):
            index = newest + offset
            frames.append(
                self._first if index < 0 else self._frames[index - oldest_held]
            )
        return {key: np.stack([f[key] for f in frames]) for key in frames[-1]}


class ReplanSchedule:
    """Plan at the first observation, then every ``replan_steps`` grid steps.

    A plan that comes due while inference is still running starts at the
    first observation after it finishes, so the interval never falls below
    ``replan_steps`` and latency is reported rather than queued.
    """

    def __init__(self, replan_steps: int):
        if replan_steps <= 0:
            raise ValueError("replan_steps must be positive")
        self.replan_steps = replan_steps
        self.reset()

    def reset(self) -> None:
        self.last_index: int | None = None

    def due(self, index: int) -> bool:
        return self.last_index is None or index - self.last_index >= self.replan_steps

    def planned(self, index: int) -> None:
        self.last_index = index


def _mix_quaternions(old: np.ndarray, new: np.ndarray, weight: float) -> np.ndarray:
    """Normalized linear interpolation on the shorter arc."""
    if float(np.dot(old, new)) < 0.0:
        old = -old
    mixed = (1.0 - weight) * old + weight * new
    return mixed / np.linalg.norm(mixed)


def blend_seam(
    chunk: np.ndarray,
    committed_count: int,
    replaceable: np.ndarray,
    steps: int,
) -> np.ndarray:
    """Crossfade a new wire chunk from the robot's current plan at the seam.

    ``chunk`` (1, T, 16) is grounded at the planning observation, whose
    ``committed_actions`` cover slots [0, committed_count) and whose
    ``replaceable_actions`` (1, R, 16) are the plan the chunk supersedes from
    slot committed_count on. Slot committed_count + j receives weight
    (j + 1) / (steps + 1) on the new chunk; later slots are unchanged.
    """
    out = np.array(chunk, dtype=np.float32, copy=True)
    if steps <= 0 or replaceable.size == 0:
        return out
    count = min(steps, replaceable.shape[1], out.shape[1] - committed_count)
    for j in range(max(count, 0)):
        weight = (j + 1) / (steps + 1)
        old = np.asarray(replaceable[0, j], dtype=np.float64)
        new = out[0, committed_count + j].astype(np.float64)
        mixed = (1.0 - weight) * old + weight * new
        for sl in _WIRE_QUAT:
            mixed[sl] = _mix_quaternions(old[sl], new[sl], weight)
        out[0, committed_count + j] = mixed
    return out
