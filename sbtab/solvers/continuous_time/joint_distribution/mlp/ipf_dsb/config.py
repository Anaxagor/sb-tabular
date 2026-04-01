from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Optional

@dataclass
class IPFDSBConfig:
    """Solver configuration IPF-DSB."""
    ipf_iters: int = 6

    num_steps: int = 20
    gamma_min: float = 1e-4
    gamma_max: float = 1e-2
    schedule: Literal["linear", "geom"] = "geom"

    batch_size: int = 512
    cache_batches: int = 200  
    lr: float = 2e-4
    weight_decay: float = 0.0
    epochs_per_phase: int = 1
    grad_clip: Optional[float] = 1.0

    noise: bool = True
    device: str = "cpu"
    seed: int = 42