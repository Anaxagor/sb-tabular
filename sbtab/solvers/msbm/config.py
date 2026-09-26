from dataclasses import dataclass
import math
from typing import Optional, Tuple, Literal

FB = Literal["f", "b"]

@dataclass
class MixedSBMConfig:
    # IMF stage sequence. Orientation: x0 = data (t = 0), x1 = prior (t = 1);
    # 'f' trains the data -> prior drift, 'b' the prior -> data drift used for
    # generation. Directions must alternate: a stage is always trained on the
    # coupling produced by the OPPOSITE direction's latest model.
    fb_sequence: Tuple[FB, ...] = ("b", "f", "b", "f", "b")

    cat_emb_dim: int = 16
    hidden_dim: int = 512
    time_dim: int = 128
    n_layers: int = 5
    dropout: float = 0.1

    # One uniform unit-horizon grid (t[n] = n / num_steps) drives the Brownian
    # bridge, the categorical reference, the network clock and the sampler.
    num_steps: int = 100
    sigma: float = 0.1
    lambda_num: float = 0.8
    lambda_cat: float = 0.2
    ce_lambda: float = 0.001

    # Categorical reference Q(s) = exp(s R); see sbtab.bridge.reference.
    cat_mixing_rate: float = 1.0
    cat_ordered_bandwidth: float = 0.2

    lr: float = 1e-4
    batch_size: int = 256
    epochs_per_direction: int = 5
    grad_clip: Optional[float] = 1.0
    # True: each direction keeps its own network across stages (warm start).
    # False: a direction's network is re-initialised at the start of every stage.
    warm_start: bool = True

    device: str = "cpu"
    seed: int = 42

    def __post_init__(self):
        for name in ("epochs_per_direction", "batch_size", "num_steps"):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) != value or value < 1:
                raise ValueError(f"{name} must be an integer >= 1")
        for name in ("lr", "sigma"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
