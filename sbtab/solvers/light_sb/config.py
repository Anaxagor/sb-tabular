from __future__ import annotations

from dataclasses import dataclass, field

from sbtab.models.sb.light_sb import LightSBPotentialConfig


@dataclass
class LightSBConfig:
    """
    Solver config for LightSB.

    Training minimizes the empirical LightSB objective:
        E_{x0 ~ p0}[log C_theta(x0)] - E_{x1 ~ p1}[log v_theta(x1)]

    where p0 is the Gaussian reference and p1 is the data distribution.
    """

    potential: LightSBPotentialConfig = field(default_factory=LightSBPotentialConfig)

    lr: float = 1e-2
    weight_decay: float = 0.0
    batch_size: int = 256
    max_iter: int = 10_000
    grad_clip: float | None = None

    init_r_from_data: bool = True

    # Sampling options
    use_sde_sampling: bool = False
    n_euler_steps: int = 100

    device: str = "cpu"
    seed: int = 42
    verbose_every: int = 1000
