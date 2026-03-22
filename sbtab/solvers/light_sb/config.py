from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LightSBConfig:
    n_potentials: int = 50
    epsilon: float = 0.1
    is_diagonal: bool = True
    S_diagonal_init: float = 1.0
    sampling_batch_size: int = 512

    max_iter: int = 10_000
    batch_size: int = 512
    lr: float = 1e-3
    safe_t: float = 1e-2
    init_r_from_data: bool = True

    use_sde_sampling: bool = False
    n_euler_steps: int = 100

    device: str = "cpu"
    seed: int = 42
