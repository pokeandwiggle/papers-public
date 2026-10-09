"""Diffusion Policy with strided observation history and server-paced replanning.

Implements the context-length / execution-horizon study of Zeng et al.,
"Revisiting Open-Loop Execution in Robotics" (arXiv:2608.15938) on the local
LeRobot Diffusion backbone. Importing this package registers the policy type.
"""

from paw_baseline.hist_diffusion import (  # noqa: F401
    configuration_hist_diffusion,
    processor_hist_diffusion,
)
