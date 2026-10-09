"""The authors' real-robot Diffusion Policy (arXiv:2608.15938) on this platform.

DINOv3 double-encoder cross-attention U-Net with strided history, past-token
prediction, per-step delta actions, EMA and server-paced replanning.
Importing this package registers the ``attn_diffusion`` policy type.
"""

from paw_baseline.attn_diffusion import (  # noqa: F401
    configuration_attn_diffusion,
    processor_attn_diffusion,
)
