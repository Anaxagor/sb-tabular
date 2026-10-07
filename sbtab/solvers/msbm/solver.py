from __future__ import annotations
from dataclasses import asdict
from typing import List, Optional, Dict, Any, Tuple, Literal
import torch
from tqdm import tqdm
from .pathsampler import MixedPathSampler
from sbtab.bridge.reference import GaussianReference
from .reference import CategoricalReference
from sbtab.bridge.sde import EulerMaruyama
from sbtab.bridge.timegrid import TimeGrid
from sbtab.models.neural.MixedMLP import MixedSbmMlp
from .updater import MixedSBMUpdater
from .config import MixedSBMConfig

CHECKPOINT_FORMAT = "sbtab.mixedsbm/3"


class MixedSBMSolver:
    def __init__(
        self,
        continuous_dim: int,
        cardinalities: List[int],
        is_ordered: Optional[torch.Tensor],
        cfg: "MixedSBMConfig"
    ) -> None:
        """
        MSBM solver init.

        Args:
            continuous_dim (int): Dimension of the continuous part of the data.
            cardinalities (List[int]): List of cardinalities (number of possible values) of each category.
            is_ordered (Optional[torch.Tensor]): Boolean torch Tensor (used as a mask) where True if the category is ordered otherwise False.
            cfg (MixedSBMConfig): MSBM configuration object.

        Raises:
            ValueError: If there's no data to use.

        Returns:
            None
        """
        self.cont_dim = continuous_dim
        self.cardinalities = [int(c) for c in cardinalities]
        is_ordered = torch.as_tensor(is_ordered if is_ordered is not None else [False] * len(cardinalities), dtype=torch.bool)
        self.is_ordered = is_ordered
        if len(is_ordered) != len(cardinalities) or any(c < 1 for c in cardinalities):
            raise ValueError("invalid cardinalities or is_ordered mask")
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        torch.manual_seed(cfg.seed)

        self.has_cont = continuous_dim > 0
        self.has_cat = len(cardinalities) > 0

        if not self.has_cont and not self.has_cat:
            raise ValueError("There is no data to process.")

        self.ref_gauss = GaussianReference(
            dim=continuous_dim,
            mean=cfg.num_ref_mean,
            std=cfg.num_ref_std,
            device=self.device,
            dtype=cfg.num_dtype
        ) if self.has_cont else None

        self.ref_cat = CategoricalReference(
            cardinalities=cardinalities,
            is_ordered=is_ordered,
            total_number_of_q_powers=cfg.num_steps,
            alpha=cfg.alpha,
            device=self.device,
            dtype=cfg.cat_dtype
        ) if self.has_cat else None

        self.integrator = EulerMaruyama(noise=cfg.noise, sigma=cfg.sigma) if self.has_cont else None
        self.timegrid = TimeGrid.uniform(cfg.num_steps)

        self.sampler = MixedPathSampler(
            timegrid=self.timegrid,
            reference=self.ref_cat,
            integrator=self.integrator,
            has_cont=self.has_cont,
            has_cat=self.has_cat
        )

        self.model = MixedSbmMlp(
            continuous_dim=continuous_dim,
            cardinalities=cardinalities,
            cat_emb_dim=cfg.cat_emb_dim,
            hidden_dim=cfg.hidden_dim,
            time_dim=cfg.time_dim,
            n_layers=cfg.n_layers,
            dropout=cfg.dropout,
        ).to(self.device)

        self.updater = MixedSBMUpdater(self.model, self.ref_cat, cfg, self.has_cont, self.has_cat)
        self.last_b_state = None
        self.stage_log = []
        self._fitted = False

    def _resolve_sim_batch_size(self, *tensors: Optional[torch.Tensor]) -> int:
        """
        Chunk size for path simulation.

        Args:
            tensors (Optional[torch.Tensor]): tensors which are used at the start of simulation.

        Returns:
            int: Chunk size for path simulation.
        """
        n = 0
        for t in tensors:
            if t is not None and t.numel() > 0:
                n = int(t.shape[0])
                break
        sim_bs = getattr(self.cfg, "sim_batch_size", None)
        if sim_bs is None:
            sim_bs = 100_000 if not self.has_cat else 16_384
        elif self.has_cat:
            sim_bs = min(sim_bs, 16_384)
        sim_bs = max(1, int(sim_bs))
        return min(sim_bs, n) if n > 0 else sim_bs

    @torch.no_grad()
    def _generate_coupling(
        self,
        data_num: torch.Tensor | None,
        data_cat: torch.Tensor | None,
        prior_num: torch.Tensor | None,
        prior_cat: torch.Tensor | None,
        prev_state: Dict[str, Any] | None,
        prev_dir: Literal["f", "b"],
        seed: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generates coupling between prior and data distributions.

        Args:
            data_num (torch.Tensor | None): Real data distribution of continuous features. If None then there are no continuous features.
            data_cat (torch.Tensor | None): Real data distribution of categorical features. If None then there are no categorical features.
            prior_num (torch.Tensor | None): Prior distribution of continuous features. if None then there are no continuous features.
            prior_cat (torch.Tensor | None): Prior distribution of categorical features. If None then there are no categorical features.
            prev_state (Dict[str, Any] | None): Previous state of the backbone model. If None then it's the first iteration.
            prev_dir (Literal["f", "b"]): Previous direction of the imf loop.
            seed (int): Random seed.

        Returns:
            start_numerical (torch.Tensor): starting distribution for continuous features.
            start_categorical (torch.Tensor): starting distribution for categorical features.
            end_numerical (torch.Tensor): ending distribution for continuous features.
            end_categorical (torch.Tensor): ending distribution for categorical features.
        """
        if prev_state is None:
            return data_num, data_cat, prior_num, prior_cat

        orig_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
        was_training = self.model.training

        try:
            self.model.load_state_dict(prev_state)
            self.model.eval()

            if prev_dir == 'f':
                start_num = data_num.to(self.device) if self.has_cont else data_num
                start_cat = data_cat.to(self.device) if self.has_cat else data_cat
            else:
                start_num = prior_num.to(self.device) if self.has_cont else prior_num
                start_cat = prior_cat.to(self.device) if self.has_cat else prior_cat

            end_num, end_cat, _ = self.sampler.simulate(
                x_cont_init=start_num,
                x_cat_init=start_cat,
                model=self.model,
                direction=prev_dir,
                seed=seed,
                batch_size=self._resolve_sim_batch_size(start_num, start_cat),
            )
        finally:
            self.model.load_state_dict(orig_state)
            self.model.train(was_training)

        if prev_dir == 'f':
            return start_num, start_cat, end_num, end_cat
        else:
            return end_num, end_cat, start_num, start_cat

    def _train_direction(
        self,
        direction: Literal["f", "b"],
        z0_num: torch.Tensor | None,
        z0_cat: torch.Tensor | None,
        z1_num: torch.Tensor | None,
        z1_cat: torch.Tensor | None
    ) -> Tuple[int, float]:
        """
        Calls updater to train forward or backward direction part of the loop.
        Budget is resolved inside MixedSBMUpdater.train_epochs from cfg
        (steps_per_direction / epochs_per_direction) and floored by cfg.min_steps_per_direction.

        Args:
            direction (Literal["f", "b"]): Direction of the imf loop.
            z0_num (torch.Tensor | None): Initial distribution of continuous features.
            z0_cat (torch.Tensor | None): Initial distribution of categorical features.
            z1_num (torch.Tensor | None): Ending distribution of continuous features.
            z1_cat (torch.Tensor | None): Ending distribution of categorical features.

        Returns:
            None
        """
        return self.updater.train_epochs(
            direction, z0_num, z0_cat, z1_num, z1_cat,
            min_steps=self.cfg.min_steps_per_direction
        )

    def fit(self, train_num, train_cat=None) -> MixedSBMSolver:
        self._fitted = False
        devices = [self.device] if self.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(self.cfg.seed + 104729)
            return self._fit(train_num, train_cat)

    def _fit(self, train_num, train_cat=None) -> MixedSBMSolver:
        """
        Fits MSBM solver.

        Args:
            train_num (torch.Tensor | None): Train distribution of continuous features. None if no continuous features.
            train_cat (torch.Tensor | None): Train distribution of categorical features. None if no categorical features.

        Raises:
            ValueError: If train_num and train_cat are empty at the same time.

        Returns:
            MixedSBMSolver: fitted MSBM solver.
        """
        if train_num is not None:
            train_num = torch.as_tensor(train_num, dtype=self.cfg.num_dtype, device=self.device)
            if train_num.ndim != 2 or train_num.shape[1] != self.cont_dim or not torch.isfinite(train_num).all():
                raise ValueError("invalid numerical training block")
        if train_cat is not None:
            train_cat = torch.as_tensor(train_cat, device=self.device)
            if not torch.isfinite(train_cat).all() or (train_cat.is_floating_point() and not torch.equal(train_cat, train_cat.trunc())):
                raise ValueError("categorical codes must be finite integers")
            train_cat = train_cat.long()
            if train_cat.ndim != 2 or train_cat.shape[1] != len(self.cardinalities):
                raise ValueError("invalid categorical training block")
            for i, c in enumerate(self.cardinalities):
                if train_cat.shape[0] and ((train_cat[:, i] < 0).any() or (train_cat[:, i] >= c).any()):
                    raise ValueError("categorical codes outside fitted support")
        if train_num is not None and train_cat is not None and len(train_num) != len(train_cat):
            raise ValueError("training blocks must have the same number of rows")
        if (self.has_cont and train_num is None) or (self.has_cat and train_cat is None):
            raise ValueError("missing required training block")

        if train_num is not None and train_num.numel() > 0:
            N = train_num.shape[0]
        elif train_cat is not None and train_cat.numel() > 0:
            N = train_cat.shape[0]
        else:
            raise ValueError("Train data is empty.")

        if train_num is not None:
            train_num = train_num.to(self.device)
        if train_cat is not None:
            train_cat = train_cat.to(self.device)

        if train_num is None or train_num.numel() == 0:
            train_num = torch.empty((N, 0), device=self.device)
        if train_cat is None or train_cat.numel() == 0:
            train_cat = torch.empty((N, 0), dtype=torch.long, device=self.device)

        prior_num = (
            self.ref_gauss.sample(N, seed=self.cfg.seed + 999).to(self.device)
            if self.has_cont
            else torch.empty((N, 0), device=self.device)
        )
        prior_cat = (
            torch.stack(
                [
                    torch.randint(0, c, (N,), device=self.device)
                    for c in self.cardinalities
                ],
                dim=1,
            )
            if self.has_cat
            else torch.empty((N, 0), dtype=torch.long, device=self.device)
        )

        self.stage_log = []
        self.last_b_state = None
        prev_state = None
        prev_dir = self.cfg.fb_sequence[0]
        total_stages = len(self.cfg.fb_sequence)

        with tqdm(
                total=total_stages, desc="MSBM iterations", unit="stage"
        ) as outer_pbar:
            for idx, direction in enumerate(self.cfg.fb_sequence):
                outer_pbar.set_postfix(stage=f"{direction} {idx + 1}/{total_stages}")

                z0_num, z0_cat, z1_num, z1_cat = self._generate_coupling(
                    train_num,
                    train_cat,
                    prior_num,
                    prior_cat,
                    prev_state,
                    prev_dir,
                    seed=self.cfg.seed + 10000 + idx
                )

                n_updates, loss = self._train_direction(direction, z0_num, z0_cat, z1_num, z1_cat)
                source = "independent" if prev_state is None else ("forward_model" if prev_dir == "f" else "backward_model")
                self.stage_log.append({"stage": idx, "fb": direction, "coupling_source": source,
                    "anchored_endpoint": {"independent": "both", "forward_model": "data", "backward_model": "prior"}[source],
                    "n_updates": n_updates, "last_loss": loss, "last_epoch_loss": loss})

                snap_state = {
                    k: v.detach().cpu().clone()
                    for k, v in self.model.state_dict().items()
                }
                if direction == "b":
                    self.last_b_state = snap_state

                prev_state = snap_state
                prev_dir = direction

                outer_pbar.update(1)

        self._fitted = True
        return self

    @torch.no_grad()
    def sample(self, n_samples: int, seed: int | None = None, batch_size: int | None = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Samples from the fitted MSBM solver.

        Args:
            n_samples (int): Number of samples.
            seed (int | None): Random seed.

        Raises:
            RuntimeError: If the solver is not fitted or there's no backward direction snapshot found.

        Returns:
            gen_num (torch.Tensor): generated continuous features.
            gen_cat (torch.Tensor): generated categorical features.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before sample().")

        if self.last_b_state is None:
            raise RuntimeError(
                "No backward snapshot found. Ensure fb_sequence contains at least one 'b'."
            )

        if n_samples <= 0 or (batch_size is not None and batch_size < 1):
            raise ValueError("n_samples and batch_size must be positive")
        generator = torch.Generator(device=self.device)
        if seed is None:
            seed = int(torch.randint(0, 2 ** 31, ()).item())
        generator.manual_seed(int(seed))
        orig_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
        was_training = self.model.training

        try:
            self.model.load_state_dict(self.last_b_state)
            self.model.eval()

            start_num = (
                self.ref_gauss.sample(n_samples, generator=generator).to(self.device)
                if self.has_cont
                else torch.empty((n_samples, 0), device=self.device)
            )
            start_cat = (
                torch.stack(
                    [
                        torch.randint(0, c, (n_samples,), device=self.device, generator=generator)
                        for c in self.cardinalities
                    ],
                    dim=1,
                )
                if self.has_cat
                else torch.empty((n_samples, 0), dtype=torch.long, device=self.device)
            )

            gen_num, gen_cat, _ = self.sampler.simulate(
                start_num,
                start_cat,
                model=self.model,
                direction="b",
                generator=generator,
                batch_size=batch_size or self._resolve_sim_batch_size(start_num, start_cat),
            )
        finally:
            self.model.load_state_dict(orig_state)
            self.model.train(was_training)

        return gen_num, gen_cat


    @property
    def n_updates(self) -> int:
        return sum(stage["n_updates"] for stage in self.stage_log)

    def generation_stage(self) -> int:
        for stage in reversed(self.stage_log):
            if stage["fb"] == "b":
                return stage["stage"]
        raise RuntimeError("no backward generation stage")

    def state_dict(self) -> dict:
        return {"format": CHECKPOINT_FORMAT, "config": asdict(self.cfg),
                "continuous_dim": self.cont_dim, "cardinalities": self.cardinalities,
                "is_ordered": self.is_ordered.tolist(), "stage_log": self.stage_log,
                "model": {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()},
                "last_b_state": self.last_b_state, "fitted": self._fitted}

    def save_checkpoint(self, path) -> None:
        torch.save(self.state_dict(), path)

    @classmethod
    def load_checkpoint(cls, path, device=None):
        state = torch.load(path, map_location="cpu", weights_only=False)
        if state.get("format") != CHECKPOINT_FORMAT:
            raise ValueError("checkpoint is not a feature/tuning MSBM checkpoint (sbtab.mixedsbm/3)")
        config = dict(state["config"])
        config["fb_sequence"] = tuple(config["fb_sequence"])
        if device is not None:
            config["device"] = device
        solver = cls(state["continuous_dim"], state["cardinalities"], state["is_ordered"], MixedSBMConfig(**config))
        solver.model.load_state_dict(state["model"])
        solver.last_b_state = state["last_b_state"]
        solver.stage_log = state["stage_log"]
        solver._fitted = bool(state["fitted"])
        return solver
