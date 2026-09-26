from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict
from typing import Optional

import numpy as np
import pandas as pd
import torch

from sbtab.bridge.reference import GaussianReference
from sbtab.models.sb.light_sb import LightSBPotential, LightSBPotentialConfig
from sbtab.solvers.light_sb.config import LightSBConfig

CHECKPOINT_FORMAT = "sbtab.lightsb.v1"


class LightSBSolver:
    """
    LightSB solver (registry id ``lightsb``). LightSB-M is NOT implemented.

    - Training uses the empirical objective from the LightSB paper / official code:
        mean(log C_theta(x0)) - mean(log v_theta(x1))
      with x0 ~ N(0, I) (reference side) and x1 ~ data.
    - Fast sampling uses the learned conditional plan pi_theta(x1 | x0).
    - Optional SDE sampling uses the exact drift with Euler-Maruyama.

    Randomness
      Nothing here reseeds the caller's global RNG. Seeded calls use ONE
      ``torch.Generator`` for the start draw x0 and for everything after it (the
      Brownian increments of the SDE sampler continue the same stream; they do not
      replay x0). The direct GMM sampler (torch.distributions) cannot take a
      generator, so seeded direct sampling runs inside ``torch.random.fork_rng()``
      with a seed derived from that generator; the global RNG state is restored.
      Unseeded calls draw x0 from the solver's own reference generator and the
      rest from the global RNG.

    Public API
      fit(train) -> self
      transport(x0, seed=None, use_sde_sampling=None, n_euler_steps=None, generator=None) -> ndarray
      sample(n, seed=None, use_sde_sampling=None, n_euler_steps=None) -> ndarray (n, dim)
      sample_df(...) -> DataFrame with the TRAINING column names
      sample_paths(n, seed=None, n_euler_steps=None) -> ndarray (n, n_steps + 1, dim)
      save_checkpoint(path) / load_checkpoint(path, device=None)
      n_updates: optimizer updates made by fit() so far (cumulative)
    """

    canonical_id = "lightsb"

    def __init__(self, dim: int, cfg: LightSBConfig):
        self.dim = int(dim)
        self.cfg = cfg

        if cfg.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(cfg.device)

        self.reference = GaussianReference(dim=self.dim, device=self.device)
        # Parameter init (random centres) is seeded locally; the global RNG is left alone.
        with self._forked_rng(int(cfg.seed)):
            self.model = LightSBPotential(dim=self.dim, cfg=cfg.potential).to(self.device)

        self._columns: Optional[list[str]] = None
        self._fitted = False
        self.n_updates = 0

        # Dedicated generator for Gaussian reference batches during training/sampling.
        self._ref_gen = self._generator(int(cfg.seed) + 1)

    # ------------------------------------------------------------------
    # utilities
    # ------------------------------------------------------------------

    @property
    def variant_id(self) -> str:
        return self.canonical_id

    @contextmanager
    def _forked_rng(self, seed: int):
        """Run a block with a locally seeded global RNG; the caller's RNG state is restored."""
        devices = [self.device] if self.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(int(seed))
            yield

    def _generator(self, seed: int) -> torch.Generator:
        gen = torch.Generator(device=str(self.device))
        gen.manual_seed(int(seed))
        return gen

    def _dtype(self) -> torch.dtype:
        return self.model.r.dtype

    def _as_tensor(self, x: pd.DataFrame | np.ndarray | torch.Tensor) -> torch.Tensor:
        # NOTE: never records column names; only fit() defines the training columns.
        if isinstance(x, pd.DataFrame):
            x = x.to_numpy(dtype=np.float32, copy=True)
        if isinstance(x, np.ndarray):
            return torch.from_numpy(np.ascontiguousarray(x)).to(self.device, dtype=self._dtype())
        if isinstance(x, torch.Tensor):
            return x.to(self.device, dtype=self._dtype())
        raise TypeError(f"Unsupported type: {type(x)}")

    def _sample_reference_batch(
        self,
        n: int,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        gen = self._ref_gen if generator is None else generator
        x = torch.randn(
            (int(n), self.dim),
            device=self.device,
            dtype=self._dtype(),
            generator=gen,
        )
        # GaussianReference currently uses mean/std scalars, so we match its distribution.
        return x * float(self.reference.std) + float(self.reference.mean)

    def _sample_data_batch(
        self,
        x_data: torch.Tensor,
        batch_size: int,
        generator: torch.Generator,
    ) -> torch.Tensor:
        N = x_data.shape[0]
        idx = torch.randint(0, N, (int(batch_size),), generator=generator, device=self.device)
        return x_data[idx]

    # ------------------------------------------------------------------
    # training
    # ------------------------------------------------------------------

    def fit(self, train: pd.DataFrame | np.ndarray | torch.Tensor) -> "LightSBSolver":
        x_data = self._as_tensor(train)
        if x_data.ndim != 2 or x_data.shape[1] != self.dim:
            raise ValueError(f"Expected train shape (N, {self.dim}), got {tuple(x_data.shape)}")
        if x_data.shape[0] < 1:
            raise ValueError("train must contain at least one row")
        if torch.isnan(x_data).any():
            raise ValueError("Input contains NaNs. Apply preprocessing first.")
        if int(self.cfg.max_iter) < 1:
            raise ValueError("max_iter must be >= 1")
        # Column names are defined by the TRAINING frame only.
        self._columns = [str(c) for c in train.columns] if isinstance(train, pd.DataFrame) else None

        # Optional init of centers r_k by data samples (reference code / paper appendix),
        # drawn from a dedicated seeded generator (not from the global RNG).
        if self.cfg.init_r_from_data:
            init_gen = self._generator(int(self.cfg.seed) + 3)
            N = x_data.shape[0]
            K = self.cfg.potential.n_potentials
            if N >= K:
                perm = torch.randperm(N, device=self.device, generator=init_gen)[:K]
                init_samples = x_data[perm].detach().clone()
            else:
                reps = (K + N - 1) // N
                repeated = x_data.repeat(reps, 1)[:K]
                noise = 0.01 * torch.randn(
                    repeated.shape, device=self.device, dtype=repeated.dtype, generator=init_gen
                )
                init_samples = (repeated + noise).detach().clone()
            self.model.init_r_by_samples(init_samples)

        opt = torch.optim.Adam(
            self.model.parameters(),
            lr=float(self.cfg.lr),
            weight_decay=float(self.cfg.weight_decay),
        )

        data_gen = self._generator(int(self.cfg.seed) + 2)

        self.model.train()
        for it in range(int(self.cfg.max_iter)):
            x1 = self._sample_data_batch(x_data, self.cfg.batch_size, data_gen)
            x0 = self._sample_reference_batch(self.cfg.batch_size)

            loss = self.model.get_log_C(x0).mean() - self.model.get_log_potential(x1).mean()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if self.cfg.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.cfg.grad_clip))
            opt.step()
            self.n_updates += 1

            if self.cfg.verbose_every > 0 and ((it + 1) % self.cfg.verbose_every == 0 or it == 0):
                print(f"[LightSB] iter={it+1}/{self.cfg.max_iter} loss={float(loss.detach().cpu()):.6f}")

        self.model.eval()
        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def transport(
        self,
        x0: pd.DataFrame | np.ndarray | torch.Tensor,
        seed: Optional[int] = None,
        use_sde_sampling: Optional[bool] = None,
        n_euler_steps: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
    ) -> np.ndarray:
        """
        Transport given input x0 to x1.

        - if use_sde_sampling=False: sample from conditional plan π_theta(.|x0)
        - if use_sde_sampling=True:  simulate the associated bridge process with EM

        ``generator`` (takes precedence over ``seed``) is the stream the noise is
        drawn from. Pass the generator that produced x0 to CONTINUE its stream;
        building a second generator from the same seed would replay x0 as the
        first Brownian increment.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before transport().")

        if use_sde_sampling is None:
            use_sde_sampling = self.cfg.use_sde_sampling
        if n_euler_steps is None:
            n_euler_steps = self.cfg.n_euler_steps

        x0_t = self._as_tensor(x0)
        if x0_t.ndim != 2 or x0_t.shape[1] != self.dim:
            raise ValueError(f"Expected x0 shape (B, {self.dim}), got {tuple(x0_t.shape)}")

        gen = generator
        if gen is None and seed is not None:
            gen = self._generator(seed)

        if use_sde_sampling:
            traj = self.model.sample_euler_maruyama(x0_t, n_steps=int(n_euler_steps), generator=gen)
            x1 = traj[:, -1, :]
        elif gen is None:
            x1 = self.model(x0_t)
        else:
            # torch.distributions cannot take a generator: seed a FORKED global RNG
            # with a value derived from (not equal to the seed of) the generator.
            derived = int(torch.randint(0, 2 ** 62, (1,), generator=gen, device=self.device).item())
            with self._forked_rng(derived):
                x1 = self.model(x0_t)

        return x1.detach().cpu().numpy()

    @torch.no_grad()
    def sample(
        self,
        n: int,
        seed: Optional[int] = None,
        use_sde_sampling: Optional[bool] = None,
        n_euler_steps: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate n samples from the target side by starting from the Gaussian reference.
        With a seed, ONE generator drives the start draw and the transport noise.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before sample().")
        if n <= 0:
            raise ValueError("n must be positive")

        generator = None if seed is None else self._generator(seed)
        x0 = self._sample_reference_batch(int(n), generator=generator)
        return self.transport(
            x0,
            use_sde_sampling=use_sde_sampling,
            n_euler_steps=n_euler_steps,
            generator=generator,
        )

    def sample_df(
        self,
        n: int,
        seed: Optional[int] = None,
        use_sde_sampling: Optional[bool] = None,
        n_euler_steps: Optional[int] = None,
    ) -> pd.DataFrame:
        arr = self.sample(
            n=n,
            seed=seed,
            use_sde_sampling=use_sde_sampling,
            n_euler_steps=n_euler_steps,
        )
        if self._columns is None:
            return pd.DataFrame(arr)
        return pd.DataFrame(arr, columns=self._columns)

    @torch.no_grad()
    def sample_paths(
        self,
        n: int,
        seed: Optional[int] = None,
        n_euler_steps: Optional[int] = None,
    ) -> np.ndarray:
        """
        Sample full Euler-Maruyama trajectories starting from the Gaussian reference.

        Returns:
            array of shape (n, n_steps + 1, dim); [:, -1] equals
            sample(n, seed, use_sde_sampling=True, n_euler_steps=n_euler_steps).
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before sample_paths().")
        if n <= 0:
            raise ValueError("n must be positive")
        if n_euler_steps is None:
            n_euler_steps = self.cfg.n_euler_steps

        generator = None if seed is None else self._generator(seed)
        x0 = self._sample_reference_batch(int(n), generator=generator)
        traj = self.model.sample_euler_maruyama(x0, n_steps=int(n_euler_steps), generator=generator)
        return traj.detach().cpu().numpy()

    # ------------------------------------------------------------------
    # checkpoint
    # ------------------------------------------------------------------

    def describe(self) -> dict:
        return {
            "solver_id": self.canonical_id,
            "variant_id": self.variant_id,
            "algorithm": "LightSB (Korotin et al. 2024); LightSB-M is not implemented",
            "epsilon": float(self.cfg.potential.epsilon),
            "n_potentials": int(self.cfg.potential.n_potentials),
            "covariance": "diagonal" if self.cfg.potential.is_diagonal else "full (geotorch)",
            "reference_side": "x0 ~ N(0, I)",
            "default_sampler": "sde_euler_maruyama" if self.cfg.use_sde_sampling else "conditional_plan",
        }

    def state_dict(self) -> dict:
        """Inference-complete state; reload never refits."""
        return {
            "format": CHECKPOINT_FORMAT,
            "solver_id": self.canonical_id,
            "dim": self.dim,
            "config": asdict(self.cfg),  # nested: config["potential"] is the potential config
            "model_state": {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()},
            "model_dtype": str(self._dtype()).replace("torch.", ""),
            "columns": None if self._columns is None else list(self._columns),
            "fitted": bool(self._fitted),
            "n_updates": int(self.n_updates),
            "ref_gen_state": self._ref_gen.get_state().cpu().clone(),
            "ref_gen_device": self.device.type,
        }

    def save_checkpoint(self, path) -> None:
        torch.save(self.state_dict(), path)

    @classmethod
    def load_checkpoint(cls, path, device: Optional[str] = None) -> "LightSBSolver":
        state = torch.load(path, map_location="cpu", weights_only=True)
        if state.get("format") != CHECKPOINT_FORMAT:
            raise ValueError(f"unsupported LightSB checkpoint format: {state.get('format')!r}")
        cfg_dict = dict(state["config"])
        potential = LightSBPotentialConfig(**cfg_dict.pop("potential"))
        if device is not None:
            cfg_dict["device"] = device
        cfg = LightSBConfig(potential=potential, **cfg_dict)

        # epsilon lives both in the config and in the model buffer; they must agree.
        stored_eps = float(state["model_state"]["epsilon"])
        expected_eps = float(torch.tensor(float(potential.epsilon), dtype=torch.float32))
        if stored_eps != expected_eps:
            raise ValueError(
                f"checkpoint epsilon buffer ({stored_eps!r}) does not match its config ({expected_eps!r})"
            )

        solver = cls(int(state["dim"]), cfg)
        solver.model.to(getattr(torch, state["model_dtype"]))
        solver.model.load_state_dict(state["model_state"])
        solver.model.eval()
        solver._columns = None if state["columns"] is None else list(state["columns"])
        solver._fitted = bool(state["fitted"])
        solver.n_updates = int(state["n_updates"])
        # Generator states are device-specific; restore only onto the same device type
        # (otherwise the reference stream restarts from cfg.seed + 1).
        if state.get("ref_gen_device") == solver.device.type:
            solver._ref_gen.set_state(state["ref_gen_state"])
        return solver
