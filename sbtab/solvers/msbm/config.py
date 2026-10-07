import math
import torch

from dataclasses import dataclass
from typing import Optional, Tuple, Literal

FB = Literal["f", "b"]

@dataclass
class MixedSBMConfig:
    """
    Configuration for MSBM solver.

    Training budget per IMF direction is resolved in MixedSBMUpdater.train_epochs:
        1. explicit `steps=` argument;
        2. explicit `epochs=` argument;
        3. `steps_per_direction` (fixed optimizer steps, N-independent, DSBM inner_iters semantics);
        4. `epochs_per_direction` * ceil(N / batch_size);
        5. none of the above -> ValueError.
    The resolved budget is floored by `min_steps_per_direction` (0 disables the floor).

    Attributes:
        fb_sequence: Sequence of phases "forward" and "backward" encoded as "f" and "b". Always starts and ends with "b".
        cat_emb_dim: Dimensionality of categorical embeddings.
        hidden_dim: Hidden layer's dimensionality.
        time_dim: Time dimensionality.
        n_layers: Number of layers in MSBM.
        dropout: Dropout rate.
        weight_decay: Weight decay of the backbone model.
        num_steps: Number of steps in one outer loop of MSBM.
        sigma: Sigma parameter of Euler-Maruyama integrator.
        alpha: Alpha parameter of Categorical reference.
        lambda_num: Weight of the loss for continuous features.
        lambda_cat: Weight of the loss for categorical features.
        ce_lambda: Scaling weight for the auxiliary cross-entropy inside CSBM.
        num_ref_mean: Mean value of the numerical reference process.
        num_ref_std: Standard deviation of the numerical reference process.
        noise: Euler-Maruyama noise parameter.
        lr: Learning rate.
        batch_size: Training batch size.
        sim_batch_size: Chunk size for path simulation (`_generate_coupling`, `sample`).
            Memory-guard only, independent of batch_size; capped at 16_384 when categorical
            features are present (large [B, D, S_max, S_max] gathers in CategoricalReference).
        epochs_per_direction: Epochs per IMF direction. Used as budget when `steps_per_direction`
            is None: total steps = epochs_per_direction * ceil(N / batch_size).
        steps_per_direction: Fixed optimizer steps per IMF direction. Takes priority over
            `epochs_per_direction`.
        min_steps_per_direction: Floor in steps applied to the resolved per-direction budget.
            0 disables the floor.
        grad_clip: Gradient clipping parameter.
        cat_dtype: torch.dtype of categorical reference process.
        num_dtype: torch.dtype of continuous reference process.
        device: Device to run the model on.
        seed: Random seed.
    """
    fb_sequence: Tuple[FB, ...] = ("b", "f", "b", "f", "b")

    cat_emb_dim: int = 16
    hidden_dim: int = 512
    time_dim: int = 128
    n_layers: int = 5
    dropout: float = 0.1
    weight_decay: float = 1e-2

    num_steps: int = 100
    sigma: float = 0.1
    alpha: float = 0.01
    lambda_num: float = 0.8
    lambda_cat: float = 0.2
    ce_lambda: float = 0.001
    num_ref_mean: float = 0.0
    num_ref_std: float = 1.0
    noise: bool = True

    lr: float = 1e-4
    batch_size: int = 256
    sim_batch_size: Optional[int] = 100_000
    epochs_per_direction: Optional[int] = None
    steps_per_direction: Optional[int] = 300
    min_steps_per_direction: int = 300
    grad_clip: Optional[float] = 1.0

    cat_dtype: torch.dtype = torch.float32
    num_dtype: torch.dtype = torch.float32
    device: str = "cpu"
    seed: int = 42

    def __post_init__(self):
        for name in ("num_steps", "batch_size", "hidden_dim", "n_layers", "cat_emb_dim", "time_dim"):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) != value or value < 1:
                raise ValueError(f"{name} must be an integer >= 1")
        for name in ("epochs_per_direction", "steps_per_direction", "sim_batch_size"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or int(value) != value or value < 1):
                raise ValueError(f"{name} must be an integer >= 1 or None")
        if self.steps_per_direction is None and self.epochs_per_direction is None:
            raise ValueError("set steps_per_direction or epochs_per_direction")
        if int(self.min_steps_per_direction) != self.min_steps_per_direction or self.min_steps_per_direction < 0:
            raise ValueError("min_steps_per_direction must be an integer >= 0")
        for name in ("lr", "sigma", "num_ref_std"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.alpha) or not 0 < self.alpha <= 1:
            raise ValueError("alpha must be in (0, 1]")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and nonnegative")
        if not self.fb_sequence or any(d not in ("f", "b") for d in self.fb_sequence):
            raise ValueError("fb_sequence must contain directions f or b")
        if any(a == b for a, b in zip(self.fb_sequence, self.fb_sequence[1:])):
            raise ValueError("fb_sequence must alternate")
