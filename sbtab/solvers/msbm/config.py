from dataclasses import dataclass
from typing import Optional, Tuple, Literal

FB = Literal["f", "b"]

@dataclass
class MixedSBMConfig:
    fb_sequence: Tuple[FB, ...] = ("b", "f", "b", "f", "b")

    cat_emb_dim: int = 16
    hidden_dim: int = 512
    time_dim: int = 128
    n_layers: int = 5
    dropout: float = 0.1

    num_steps: int = 100
    sigma: float = 0.1
    alpha: float = 0.01
    lambda_num: float = 0.8
    lambda_cat: float = 0.2
    eps: float = 1e-3

    lr: float = 1e-4
    batch_size: int = 256
    epochs_per_direction: int = 5
    grad_clip: Optional[float] = 1.0

    device: str = "cpu"
    seed: int = 42