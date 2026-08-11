"""Clean tensor boundary for the repository's native TabDDPM implementation.

The solver accepts one ``float32`` numerical tensor and one ``int64`` state
tensor. ``cardinalities`` describes the state columns in tensor order. The
solver converts state indices to the one-hot log representation required by
``GaussianMultinomialDiffusion``; it does not discover schemas, fit tabular
preprocessing, or reconstruct raw values.

The implementation deliberately retains the existing denoiser, Gaussian and
multinomial losses, beta schedule, AdamW optimizer, fixed-step linear learning
rate annealing, post-step EMA update, and ancestral sampler. This module only
separates those native operations from the legacy DataFrame wrapper.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from sbtab.baselines.tabddpm.gaussian_multinomial_diffsuion import (
    GaussianMultinomialDiffusion,
)
from sbtab.baselines.tabddpm.modules import MLPDiffusion


@dataclass
class TabDDPMConfig:
    """Native training and sampling controls for one TabDDPM instance.

    Parameters
    ----------
    steps:
        Number of optimizer updates. When set, this takes precedence over
        ``n_epochs`` and preserves the current fixed-step training behavior.
    n_epochs:
        Backward-compatible epoch count used only when ``steps`` is ``None``.
    num_timesteps:
        Number of forward/reverse diffusion timesteps in the native schedule.
    batch_size:
        Training mini-batch size and ancestral-sampling batch size.
    lr:
        Initial AdamW learning rate before linear step-wise annealing.
    weight_decay:
        AdamW weight-decay coefficient.
    d_layers:
        Hidden-layer widths of the native MLP denoiser.
    dropout:
        Dropout probability used in every denoiser hidden layer.
    gaussian_loss_type:
        Native numerical loss, either ``"mse"`` (the existing default) or
        ``"kl"``. The value is passed to the diffusion instead of being
        silently ignored.
    scheduler:
        Native beta schedule name, either ``"cosine"`` or ``"linear"``.
        The inherited linear formula requires more than 20 timesteps so its
        largest beta remains strictly below one.
    ema_decay:
        Post-optimizer-step exponential moving-average decay for the denoiser.
    device:
        Torch device on which training and diffusion sampling execute.
    seed:
        Non-negative 32-bit seed controlling denoiser initialization,
        DataLoader shuffling, diffusion timesteps, and training noise.
    use_ema_for_sampling:
        Whether the native ancestral sampler uses the EMA denoiser. This is a
        typed model setting; the shared adapter API has no model-specific
        keyword arguments. It follows the legacy fields so their positional
        constructor slots remain unchanged.
    show_progress:
        Whether native training and sampling render progress bars. It is an
        operational control only and does not change model mathematics.
    """

    steps: int | None = 10_000
    n_epochs: int | None = None
    num_timesteps: int = 1_000
    batch_size: int = 4_096
    lr: float = 1e-3
    weight_decay: float = 1e-4
    d_layers: list[int] = field(
        default_factory=lambda: [256, 512, 512, 256]
    )
    dropout: float = 0.0
    gaussian_loss_type: str = "mse"
    scheduler: str = "cosine"
    ema_decay: float = 0.999
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    use_ema_for_sampling: bool = True
    show_progress: bool = False


def _validate_seed(seed: int, field_name: str) -> None:
    """Validate a seed accepted by both NumPy and the benchmark contract."""

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(f"{field_name} must be an integer.")
    if not 0 <= seed < 2**32:
        raise ValueError(f"{field_name} must be in the range [0, 2**32).")


def _validate_config(config: TabDDPMConfig) -> None:
    """Reject only settings that otherwise fail opaquely in native code."""

    if config.steps is None and config.n_epochs is None:
        raise ValueError(
            "TabDDPMConfig requires steps or n_epochs for native training."
        )
    active_length = config.steps if config.steps is not None else config.n_epochs
    if active_length is None or active_length <= 0:
        raise ValueError("TabDDPM training length must be positive.")
    if config.num_timesteps < 2:
        raise ValueError(
            "TabDDPMConfig.num_timesteps must be at least 2 for the native "
            "posterior variance schedule."
        )
    if config.scheduler == "linear" and config.num_timesteps <= 20:
        raise ValueError(
            "TabDDPM linear schedule requires num_timesteps > 20 so every "
            "beta remains strictly below 1."
        )
    if config.batch_size <= 0:
        raise ValueError("TabDDPMConfig.batch_size must be positive.")
    if config.gaussian_loss_type not in {"mse", "kl"}:
        raise ValueError(
            "TabDDPMConfig.gaussian_loss_type must be 'mse' or 'kl'; "
            f"got {config.gaussian_loss_type!r}."
        )
    _validate_seed(config.seed, "TabDDPMConfig.seed")


def _seed_native_randomness(seed: int, device: torch.device) -> None:
    """Seed every RNG used by current native training and sampling code."""

    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _is_compatible_device(
    actual: torch.device,
    expected: torch.device,
) -> bool:
    """Return whether a tensor device satisfies the configured device.

    Torch may resolve an unindexed device such as ``mps`` to ``mps:0`` when a
    tensor is allocated.  An omitted index means "the current device", so it
    must not be compared strictly with the resolved tensor index.  An index
    explicitly supplied by the caller remains part of the validation.
    """

    return actual.type == expected.type and (
        expected.index is None or actual.index == expected.index
    )


class TabDDPMSolver:
    """Train and sample native TabDDPM from already prepared tensors.

    ``num_numerical_features`` is the width of the Gaussian block.
    ``cardinalities`` contains the real train-observed cardinality of every
    multinomial state column in native tensor order. The solver does not use
    ordinal adjacency: all states within one column are symmetric.
    """

    def __init__(
        self,
        num_numerical_features: int,
        cardinalities: list[int],
        cfg: TabDDPMConfig,
    ) -> None:
        _validate_config(cfg)
        if num_numerical_features < 0:
            raise ValueError("num_numerical_features must be non-negative.")
        if any(value <= 0 for value in cardinalities):
            raise ValueError("Every state cardinality must be positive.")
        if num_numerical_features == 0 and not cardinalities:
            raise ValueError("TabDDPM requires at least one modeled column.")

        self.num_numerical_features = num_numerical_features
        self.cardinalities = list(cardinalities)
        self.cfg = copy.deepcopy(cfg)
        self.device = torch.device(cfg.device)
        self.diffusion: GaussianMultinomialDiffusion | None = None
        self.ema_model: torch.nn.Module | None = None

    @staticmethod
    def _anneal_lr(
        optimizer: torch.optim.Optimizer,
        *,
        init_lr: float,
        step: int,
        total_steps: int,
    ) -> None:
        """Apply the existing linear learning-rate schedule by optimizer step."""

        fraction_complete = step / float(total_steps)
        learning_rate = init_lr * (1.0 - fraction_complete)
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = learning_rate

    @staticmethod
    @torch.no_grad()
    def _update_ema(
        target_model: torch.nn.Module,
        source_model: torch.nn.Module,
        rate: float,
    ) -> None:
        """Update the denoiser EMA immediately after one optimizer step."""

        for target, source in zip(
            target_model.parameters(), source_model.parameters()
        ):
            target.detach().mul_(rate).add_(
                source.detach(), alpha=1.0 - rate
            )

    def _validate_train_tensors(
        self,
        train_num: torch.Tensor,
        train_state: torch.Tensor,
    ) -> None:
        """Validate the native tensor seam owned by this solver."""

        for value, name in (
            (train_num, "train_num"),
            (train_state, "train_state"),
        ):
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor.")
            if value.ndim != 2:
                raise ValueError(f"{name} must be a two-dimensional tensor.")
            if not _is_compatible_device(value.device, self.device):
                raise ValueError(
                    f"{name} is on {value.device}, expected {self.device}."
                )
        if train_num.dtype != torch.float32:
            raise TypeError("train_num must have torch.float32 dtype.")
        if train_state.dtype != torch.int64:
            raise TypeError("train_state must have torch.int64 dtype.")
        if train_num.shape[0] == 0:
            raise ValueError("TabDDPM training tensors must contain rows.")
        if train_num.shape[0] != train_state.shape[0]:
            raise ValueError("train_num and train_state must have equal row counts.")
        if train_num.shape[1] != self.num_numerical_features:
            raise ValueError(
                "train_num width does not match num_numerical_features: "
                f"{train_num.shape[1]} != {self.num_numerical_features}."
            )
        if train_state.shape[1] != len(self.cardinalities):
            raise ValueError(
                "train_state width does not match cardinalities: "
                f"{train_state.shape[1]} != {len(self.cardinalities)}."
            )

    def fit(
        self,
        train_num: torch.Tensor,
        train_state: torch.Tensor,
    ) -> TabDDPMSolver:
        """Fit the native mixed diffusion on one prepared training partition."""

        self._validate_train_tensors(train_num, train_state)
        _seed_native_randomness(self.cfg.seed, self.device)

        native_cardinalities = (
            np.asarray(self.cardinalities, dtype=np.int64)
            if self.cardinalities
            else np.asarray([0], dtype=np.int64)
        )
        denoiser_width = self.num_numerical_features + int(
            native_cardinalities.sum()
        )
        denoiser = MLPDiffusion(
            d_in=denoiser_width,
            num_classes=0,
            is_y_cond=False,
            rtdl_params={
                "d_layers": list(self.cfg.d_layers),
                "dropout": float(self.cfg.dropout),
            },
        ).to(self.device)
        diffusion = GaussianMultinomialDiffusion(
            num_classes=native_cardinalities,
            num_numerical_features=self.num_numerical_features,
            denoise_fn=denoiser,
            num_timesteps=self.cfg.num_timesteps,
            gaussian_loss_type=self.cfg.gaussian_loss_type,
            scheduler=self.cfg.scheduler,
            device=self.device,
        ).to(self.device)
        optimizer = torch.optim.AdamW(
            diffusion.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )

        ema_model = copy.deepcopy(diffusion._denoise_fn).to(self.device)
        ema_model.eval()
        for parameter in ema_model.parameters():
            parameter.requires_grad_(False)

        combined_train = torch.cat(
            (train_num, train_state.to(dtype=torch.float32)), dim=1
        )
        loader_generator = torch.Generator()
        loader_generator.manual_seed(self.cfg.seed)
        loader = DataLoader(
            TensorDataset(combined_train),
            batch_size=self.cfg.batch_size,
            shuffle=True,
            drop_last=False,
            generator=loader_generator,
        )
        total_steps = (
            self.cfg.steps
            if self.cfg.steps is not None
            else int(self.cfg.n_epochs) * max(len(loader), 1)
        )

        loader_iterator = iter(loader)
        diffusion.train()
        loss_report_interval = max(total_steps // 100, 1)
        with tqdm(
            range(total_steps),
            desc="Training TabDDPM",
            unit="step",
            dynamic_ncols=True,
            disable=not self.cfg.show_progress,
        ) as progress:
            for step in progress:
                try:
                    (batch,) = next(loader_iterator)
                except StopIteration:
                    loader_iterator = iter(loader)
                    (batch,) = next(loader_iterator)

                self._anneal_lr(
                    optimizer,
                    init_lr=self.cfg.lr,
                    step=step,
                    total_steps=total_steps,
                )
                loss_multinomial, loss_gaussian = diffusion.mixed_loss(
                    batch, out_dict={"y": None}
                )
                loss = loss_multinomial + loss_gaussian

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                self._update_ema(
                    ema_model,
                    diffusion._denoise_fn,
                    self.cfg.ema_decay,
                )
                if self.cfg.show_progress and (
                    step == 0
                    or (step + 1) % loss_report_interval == 0
                    or step + 1 == total_steps
                ):
                    # Reading a device scalar synchronizes accelerators. Keep
                    # loss reporting sparse so progress output does not
                    # materially slow CUDA or MPS training.
                    progress.set_postfix(
                        loss=f"{loss.detach().item():.4f}",
                        refresh=False,
                    )

        self.diffusion = diffusion
        self.ema_model = ema_model
        return self

    @torch.no_grad()
    def sample(
        self,
        n_samples: int,
        seed: int | None,
        *,
        use_ema: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return separate numerical and finite-state tensors.

        ``seed`` is explicit for the benchmark adapter. ``None`` remains
        supported solely for the legacy wrapper, where it preserves the old
        behavior of continuing from the current RNG state.
        """

        if self.diffusion is None:
            raise RuntimeError("Call TabDDPMSolver.fit() before sample().")
        if n_samples <= 0:
            raise ValueError("n_samples must be positive.")
        if seed is not None:
            _validate_seed(seed, "sample seed")
            _seed_native_randomness(seed, self.device)
        sample_with_ema = (
            self.cfg.use_ema_for_sampling if use_ema is None else use_ema
        )
        self.diffusion.eval()
        y_distribution = torch.ones(1, device=self.device)
        denoiser_backup = self.diffusion._denoise_fn
        if sample_with_ema and self.ema_model is not None:
            self.diffusion._denoise_fn = self.ema_model

        try:
            generated, _ = self.diffusion.sample_all(
                n_samples,
                self.cfg.batch_size,
                y_distribution,
                show_progress=self.cfg.show_progress,
            )
        finally:
            self.diffusion._denoise_fn = denoiser_backup

        generated_num = generated[:, : self.num_numerical_features].to(
            dtype=torch.float32
        )
        generated_state_values = generated[:, self.num_numerical_features :]
        generated_state = generated_state_values.to(dtype=torch.int64)
        if not torch.equal(
            generated_state.to(dtype=generated_state_values.dtype),
            generated_state_values,
        ):
            raise RuntimeError(
                "Native TabDDPM returned a non-integral multinomial state."
            )
        return generated_num, generated_state
