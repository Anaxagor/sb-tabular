"""Adapters for the native mixed / categorical bridge-matching solvers."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sbtab.adapters.base import ModelAdapter
from sbtab.adapters.representation import NativeMixedRepresentation
from sbtab.solvers.csbm import AnnealedCSBMConfig, CSBMConfig, CSBMSolver
from sbtab.solvers.msbm import MixedSBMConfig, MixedSBMSolver


class MixedSBMAdapter(ModelAdapter):
    """
    MixedSBM on its native representation: continuous columns in the Brownian
    block; discrete and categorical columns as finite states (ordered kernel only
    where the schema declares an order). Pure regimes are the same solver with an
    absent block.
    """
    registry_id = "mixedsbm"
    native_regimes = ("continuous", "discrete", "mixed")
    DEFAULTS = dict(
        n_stages=5, epochs_per_direction=5, steps_per_direction=None, min_steps_per_direction=0,
        num_steps=100, sigma=0.1, lambda_num=0.8, lambda_cat=0.2,
        ce_lambda=0.001, alpha=0.01, lr=1e-4, weight_decay=1e-2, batch_size=256,
        hidden_dim=512, n_layers=5, time_dim=128, cat_emb_dim=16, dropout=0.1, grad_clip=1.0,
        noise=True, num_ref_mean=0.0, num_ref_std=1.0, sim_batch_size=4096,
        sample_batch_size=4096, device="cpu",
    )

    def _solver_config(self) -> MixedSBMConfig:
        c = self.config
        n = int(c["n_stages"])
        if n < 1:
            raise ValueError("n_stages must be >= 1")
        # alternate, ending on the backward (generation) direction
        seq = tuple("b" if (n - 1 - i) % 2 == 0 else "f" for i in range(n))
        return MixedSBMConfig(
            fb_sequence=seq, cat_emb_dim=int(c["cat_emb_dim"]), hidden_dim=int(c["hidden_dim"]),
            time_dim=int(c["time_dim"]), n_layers=int(c["n_layers"]), dropout=float(c["dropout"]),
            num_steps=int(c["num_steps"]), sigma=float(c["sigma"]), lambda_num=float(c["lambda_num"]),
            lambda_cat=float(c["lambda_cat"]), ce_lambda=float(c["ce_lambda"]),
            alpha=float(c["alpha"]), weight_decay=float(c["weight_decay"]), noise=bool(c["noise"]),
            num_ref_mean=float(c["num_ref_mean"]), num_ref_std=float(c["num_ref_std"]),
            sim_batch_size=None if c["sim_batch_size"] is None else int(c["sim_batch_size"]),
            lr=float(c["lr"]), batch_size=int(c["batch_size"]),
            epochs_per_direction=None if c["epochs_per_direction"] is None else int(c["epochs_per_direction"]),
            steps_per_direction=None if c["steps_per_direction"] is None else int(c["steps_per_direction"]),
            min_steps_per_direction=int(c["min_steps_per_direction"]),
            grad_clip=None if c["grad_clip"] is None else float(c["grad_clip"]),
            device=str(c["device"]), seed=int(self.seed),
        )

    def _prepare(self, train: pd.DataFrame) -> None:
        self.rep = NativeMixedRepresentation(self.schema).fit(train)
        self._encoded = self.rep.encode(train)

    def _build_model(self) -> None:
        self.solver = MixedSBMSolver(len(self.rep.num_cols), self.rep.cardinalities,
                                     torch.tensor(self.rep.is_ordered, dtype=torch.bool), self._solver_config())

    def _fit_model(self) -> None:
        num, cat = self._encoded
        self.solver.fit(torch.from_numpy(num), torch.from_numpy(cat))

    def _generate(self, n: int, seed: int):
        return self.solver.sample(n, seed=seed, batch_size=int(self.config["sample_batch_size"]))

    def _decode(self, generated) -> pd.DataFrame:
        num, cat = generated
        df, self.decoding_report_ = self.rep.decode(num.cpu().numpy(), cat.cpu().numpy())
        return df

    @property
    def n_updates(self):
        return self.solver.n_updates

    def describe(self) -> dict:
        s = self.solver
        return {"implementation": "feature/tuning_with_exact_categorical_bridges", "networks": "shared_forward_backward",
                "orientation": {"x0": "data", "x1": "prior", "generation": "backward"},
                "reference": {"numerical": {"kind": "brownian", "sigma": s.cfg.sigma, "horizon": 1.0,
                                            "training_noise": True, "sampling_noise": s.cfg.noise, "prior_mean": s.cfg.num_ref_mean,
                                            "prior_std": s.cfg.num_ref_std},
                              "categorical": None if s.ref_cat is None else s.ref_cat.describe()},
                "grid": {"schedule": "uniform", "num_steps": s.cfg.num_steps, "horizon": 1.0},
                "stages": s.stage_log, "generation_stage": s.generation_stage(),
                "representation": self.rep.kind}

    def _state(self) -> dict:
        return {"representation": self.rep.state()}

    def _save_model(self, directory: Path) -> None:
        self.solver.save_checkpoint(directory / "model.pt")

    def _load_model(self, directory: Path, state: dict) -> None:
        self.rep = NativeMixedRepresentation.from_state(state["representation"], self.schema)
        self.solver = MixedSBMSolver.load_checkpoint(directory / "model.pt")


class CSBMAdapter(ModelAdapter):
    """CSBM: every column must have finite support (the fully discrete regime)."""
    registry_id = "csbm"
    supported_regimes = ("discrete",)
    native_regimes = ("discrete",)
    annealed = False
    DEFAULTS = dict(
        num_outer_iterations=3, epochs=15, num_steps=50, mixing_rate=1.0, ordered_bandwidth=0.2, ce_lambda=0.001,
        lr=1e-3, batch_size=264, emb_dim=16, hidden_dim=256, time_dim=64, sample_batch_size=4096, device="cpu",
        n_layers=2, dropout=0.0, forward_lr=None, backward_lr=None,
        forward_weight_decay=1e-2, backward_weight_decay=1e-2,
    )

    def _solver_config(self):
        c = {k: v for k, v in self.config.items() if k != "sample_batch_size"}
        cls = AnnealedCSBMConfig if self.annealed else CSBMConfig
        return cls(seed=int(self.seed), **c)

    def _prepare(self, train: pd.DataFrame) -> None:
        self.rep = NativeMixedRepresentation(self.schema).fit(train)
        self._encoded = self.rep.encode(train)

    def _build_model(self) -> None:
        self.solver = CSBMSolver(self.rep.cardinalities, torch.tensor(self.rep.is_ordered, dtype=torch.bool),
                                 self._solver_config())

    def _fit_model(self) -> None:
        self.solver.fit(torch.from_numpy(self._encoded[1]))

    def _generate(self, n: int, seed: int):
        return self.solver.sample(n, seed=seed, batch_size=int(self.config["sample_batch_size"]))

    def _decode(self, generated) -> pd.DataFrame:
        cat = generated.cpu().numpy()
        df, self.decoding_report_ = self.rep.decode(np.zeros((len(cat), 0)), cat)
        return df

    @property
    def n_updates(self):
        return self.solver.n_updates

    def describe(self) -> dict:
        s = self.solver
        return {"variant": s.variant, "orientation": {"x0": "data", "x1": "prior", "generation": "backward"},
                "architecture": {"n_layers": s.cfg.n_layers, "dropout": s.cfg.dropout,
                                 "emb_dim": s.cfg.emb_dim, "hidden_dim": s.cfg.hidden_dim,
                                 "time_dim": s.cfg.time_dim},
                "optimizers": {
                    "forward": {"lr": s.updater.forward_opt.param_groups[0]["lr"],
                                "weight_decay": s.updater.forward_opt.param_groups[0]["weight_decay"]},
                    "backward": {"lr": s.updater.backward_opt.param_groups[0]["lr"],
                                 "weight_decay": s.updater.backward_opt.param_groups[0]["weight_decay"]}},
                "reference": s.reference.describe(), "reference_trace": s.reference_trace,
                "grid": {"schedule": "uniform", "num_steps": s.cfg.num_steps, "horizon": 1.0},
                "stages": s.stage_log, "representation": self.rep.kind,
                "approximation": "endpoint head factorised over columns"}

    def _state(self) -> dict:
        return {"representation": self.rep.state()}

    def _save_model(self, directory: Path) -> None:
        self.solver.save_checkpoint(directory / "model.pt")

    def _load_model(self, directory: Path, state: dict) -> None:
        self.rep = NativeMixedRepresentation.from_state(state["representation"], self.schema)
        self.solver = CSBMSolver.load_checkpoint(directory / "model.pt")


class AnnealedCSBMAdapter(CSBMAdapter):
    """Continuation heuristic; NOT the canonical fixed-reference CSBM."""
    registry_id = "csbm_annealed"
    annealed = True
    DEFAULTS = dict(CSBMAdapter.DEFAULTS, anneal_every=5, anneal_multiplier=0.9, min_mixing_rate=None)
