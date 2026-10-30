from dataclasses import dataclass, field
import math
from numbers import Integral, Real
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
    # A direction-specific value overrides lr; None keeps the legacy fallback.
    lr: float = 1e-3
    forward_lr: Optional[float] = field(default=None, kw_only=True)
    backward_lr: Optional[float] = field(default=None, kw_only=True)
    forward_weight_decay: float = field(default=1e-2, kw_only=True)
    backward_weight_decay: float = field(default=1e-2, kw_only=True)

    # One uniform unit-horizon grid: state index n lives at t[n] = n / num_steps.
    num_steps: int = 50
    mixing_rate: float = 1.0
    ordered_bandwidth: float = 0.2
    ce_lambda: float = 0.001

    emb_dim: int = 16
    hidden_dim: int = 256
    time_dim: int = 64
    n_layers: int = field(default=2, kw_only=True)
    dropout: float = field(default=0.0, kw_only=True)

    device: str = "cpu"
    seed: int = 42

    def __post_init__(self):
        for name in ("num_outer_iterations", "epochs", "batch_size", "num_steps"):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) != value or value < 1:
                raise ValueError(f"{name} must be an integer >= 1")
        for name in ("lr", "forward_lr", "backward_lr"):
            value = getattr(self, name)
            if name != "lr" and value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("forward_weight_decay", "backward_weight_decay"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if isinstance(self.n_layers, bool) or not isinstance(self.n_layers, Integral) or self.n_layers < 1:
            raise ValueError("n_layers must be an integer >= 1")
        if (isinstance(self.dropout, bool) or not isinstance(self.dropout, Real)
                or not math.isfinite(self.dropout) or not 0 <= self.dropout < 1):
            raise ValueError("dropout must be finite and in [0, 1)")

    def learning_rate(self, direction: str) -> float:
        """Effective rate: the direction override takes precedence over ``lr``."""
        if direction not in ("forward", "backward"):
            raise ValueError("direction must be forward or backward")
        override = getattr(self, f"{direction}_lr")
        return self.lr if override is None else override


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
