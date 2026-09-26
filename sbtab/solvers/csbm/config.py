from dataclasses import dataclass
import math
from typing import Optional


@dataclass
class CSBMConfig:
    """
    Canonical CSBM (D-IMF). The reference process is FIXED for the whole fit.

    Reference annealing is not part of the canonical algorithm: it changes the
    Schrödinger problem being solved between outer iterations. It is available
    only through the explicitly named continuation variant AnnealedCSBMConfig.
    """
    num_outer_iterations: int = 3
    epochs: int = 15
    batch_size: int = 264
    lr: float = 1e-3

    # One uniform unit-horizon grid: state index n lives at t[n] = n / num_steps.
    num_steps: int = 50
    mixing_rate: float = 1.0
    ordered_bandwidth: float = 0.2
    ce_lambda: float = 0.001

    emb_dim: int = 16
    hidden_dim: int = 256
    time_dim: int = 64

    device: str = "cpu"
    seed: int = 42

    def __post_init__(self):
        for name in ("num_outer_iterations", "epochs", "batch_size", "num_steps"):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) != value or value < 1:
                raise ValueError(f"{name} must be an integer >= 1")
        if not math.isfinite(self.lr) or self.lr <= 0:
            raise ValueError("lr must be finite and positive")


@dataclass
class AnnealedCSBMConfig(CSBMConfig):
    """
    Continuation heuristic (variant id ``csbm_annealed``): every ``anneal_every``
    outer iterations the mixing rate is multiplied by ``anneal_multiplier`` and
    every cached transition is rebuilt. The final model solves the bridge for the
    LAST reference only; earlier stages act as a warm start.
    """
    anneal_every: int = 5
    anneal_multiplier: float = 0.9
    min_mixing_rate: Optional[float] = None

    def __post_init__(self):
        super().__post_init__()
        if self.anneal_every < 1 or int(self.anneal_every) != self.anneal_every:
            raise ValueError("anneal_every must be an integer >= 1")
        if not math.isfinite(self.anneal_multiplier) or self.anneal_multiplier <= 0:
            raise ValueError("anneal_multiplier must be finite and positive")
