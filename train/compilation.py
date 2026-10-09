"""Compiler assumptions for LeRobot's forward-only AMP context."""

from contextlib import contextmanager


@contextmanager
def training_compilation(model: str | None):
    """Match PI05's compiled backward to LeRobot's backward outside autocast.

    Keep the context active through lazy forward/backward compilation. Other
    families retain their compiler configuration. PyTorch documents this contract:
    https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler_backward.html
    """
    if model != "paw_pi05":
        yield
        return
    from torch._functorch import config
    from torch._inductor import config as inductor

    # Preserve the BF16 rounding between operations that eager execution uses.
    # This also avoids the much larger compiled backward allocation on this model.
    with (
        config.patch(backward_pass_autocast="off"),
        inductor.patch(emulate_precision_casts=True),
    ):
        yield


@contextmanager
def training_policy_precision(trainer, model: str | None):
    """Keep Diffusion FP32 matmuls at highest precision after trainer setup.

    LeRobot enables TF32 before calling its policy factory. Apply the precision
    after that setup and before the first lazy compiled forward/backward.
    """
    if model not in ("diffusion", "hist_diffusion", "attn_diffusion"):
        yield
        return
    import torch

    original = trainer.make_policy
    precision = torch.get_float32_matmul_precision()

    def make_policy(*args, **kwargs):
        policy = original(*args, **kwargs)
        torch.set_float32_matmul_precision("highest")
        return policy

    trainer.make_policy = make_policy
    try:
        yield
    finally:
        trainer.make_policy = original
        torch.set_float32_matmul_precision(precision)
