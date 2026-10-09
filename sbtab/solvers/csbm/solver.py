from dataclasses import asdict
from typing import List, Optional

import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from sbtab.bridge.losses import CSBMLoss
from sbtab.bridge.pathsampler import DiscretePathSampler
from sbtab.bridge.reference import CategoricalReference
from sbtab.bridge.timegrid import TimeGrid
from sbtab.models.neural.CSBMTableMLP import CSBMTableMLP
from sbtab.solvers.csbm.config import AnnealedCSBMConfig, CSBMConfig
from sbtab.solvers.csbm.updater import CSBMUpdater

CHECKPOINT_FORMAT = "sbtab.csbm/2"


class CSBMSolver:
    """
    Categorical Schrödinger bridge matching (Ksenofontov & Korotin) by D-IMF.

    Orientation: x0 = data at state index 0, x1 = prior at index N; the prior is
    uniform over each column's support. The forward network predicts x_N from
    x_n (n in [0, N-1]); the backward network predicts x_0 from x_n (n in [1, N]).
    Generation runs the BACKWARD network from the prior, n = N..1, and uses the
    backward network of the last completed outer iteration.

    D-IMF couplings
      l = 1 forward    independent coupling  data (x) prior      (declared initialiser)
      l >= 2 forward   (backward-simulated x0, real prior x1)   anchored on the prior
      every backward   (real data x0, forward-simulated x1)     anchored on the data

    The reference is fixed for the whole fit unless cfg is an AnnealedCSBMConfig.
    Each endpoint head is factorised over columns (the paper's approximation):
    one step need not reproduce same-step cross-column correlation; dependence
    is built up across the N steps.
    """

    variant = "csbm"

    def __init__(self, cardinalities: List[int], is_ordered, cfg: CSBMConfig):
        self.cardinalities = [int(c) for c in cardinalities]
        if len(self.cardinalities) == 0:
            raise ValueError("CSBM needs at least one categorical column")
        self.is_ordered = torch.as_tensor(is_ordered, dtype=torch.bool).view(-1)
        self.cfg = cfg
        self.annealed = isinstance(cfg, AnnealedCSBMConfig)
        if self.annealed:
            self.variant = "csbm_annealed"
        self.device = torch.device(cfg.device)
        torch.manual_seed(cfg.seed)

        self.timegrid = TimeGrid.uniform(cfg.num_steps, horizon=1.0)
        self.reference = CategoricalReference(
            cardinalities=self.cardinalities,
            is_ordered=self.is_ordered,
            timegrid=self.timegrid,
            mixing_rate=cfg.mixing_rate,
            ordered_bandwidth=cfg.ordered_bandwidth,
            device=self.device,
        )
        self.sampler = DiscretePathSampler(timegrid=self.timegrid, reference=self.reference)
        forward_model, backward_model = self._new_model(), self._new_model()
        self.updater = CSBMUpdater(
            forward_model=forward_model,
            backward_model=backward_model,
            forward_opt=torch.optim.AdamW(forward_model.parameters(), lr=cfg.learning_rate("forward"),
                                          weight_decay=cfg.forward_weight_decay),
            backward_opt=torch.optim.AdamW(backward_model.parameters(), lr=cfg.learning_rate("backward"),
                                           weight_decay=cfg.backward_weight_decay),
            ref_process=self.reference,
            loss_fn=CSBMLoss(reference=self.reference, lmbda=cfg.ce_lambda),
            timegrid=self.timegrid,
        )
        self.stage_log = []
        self.reference_trace = []
        self._fitted = False

    def _new_model(self) -> CSBMTableMLP:
        return CSBMTableMLP(self.cardinalities, emb_dim=self.cfg.emb_dim, hidden_dim=self.cfg.hidden_dim,
                            time_dim=self.cfg.time_dim, n_layers=self.cfg.n_layers,
                            dropout=self.cfg.dropout).to(self.device)

    def _generator(self, seed: int) -> torch.Generator:
        g = torch.Generator(device=str(self.device))
        g.manual_seed(int(seed))
        return g

    def _check_codes(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.as_tensor(x, device=self.device)
        if not torch.isfinite(x).all() or (x.is_floating_point() and not torch.equal(x, x.trunc())):
            raise ValueError("categorical codes must be finite integers")
        x = x.long()
        if x.ndim != 2 or x.shape[1] != len(self.cardinalities):
            raise ValueError("expected an integer table with one column per cardinality")
        for d, c in enumerate(self.cardinalities):
            if x.shape[0] and (int(x[:, d].min()) < 0 or int(x[:, d].max()) >= c):
                raise ValueError(f"categorical column {d} has codes outside [0, {c})")
        return x

    def _train_direction(self, direction: str, x0: torch.Tensor, x1: torch.Tensor, seed: int, desc: str):
        N = self.timegrid.num_steps
        shuffle_gen = torch.Generator()
        shuffle_gen.manual_seed(int(seed))
        gen = self._generator(seed + 1)
        loader = DataLoader(TensorDataset(x0, x1), batch_size=self.cfg.batch_size, shuffle=True,
                            drop_last=False, generator=shuffle_gen)
        n_updates, last = 0, float("nan")
        for _ in range(self.cfg.epochs):
            total, batches = 0.0, 0
            pbar = tqdm(loader, desc=desc)
            for x0_batch, x1_batch in pbar:
                B = x0_batch.shape[0]
                if direction == "forward":
                    n = torch.randint(0, N, (B,), device=self.device, generator=gen)
                    xt = self.reference.sample_x_t(x0_batch, x1_batch, n, generator=gen)
                    loss = self.updater.train_forward_step(xt, x1_batch, n)
                else:
                    n = torch.randint(1, N + 1, (B,), device=self.device, generator=gen)
                    xt = self.reference.sample_x_t(x0_batch, x1_batch, n, generator=gen)
                    loss = self.updater.train_backward_step(xt, x0_batch, n)
                pbar.set_postfix(loss=f"{loss:.4f}")
                total += loss
                batches += 1
                n_updates += 1
            last = total / batches
        if self.cfg.epochs > 0 and n_updates <= 0:
            raise RuntimeError("CSBM stage performed no optimizer updates")
        return n_updates, last

    def fit(self, x_data, x_prior=None):
        """Fit with a local dropout RNG; preserve the caller's random stream."""
        self._fitted = False
        devices = [self.device] if self.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.random.default_generator.manual_seed(self.cfg.seed + 104729)
            if self.device.type == "cuda":
                with torch.cuda.device(self.device):
                    torch.cuda.manual_seed(self.cfg.seed + 104729)
            return self._fit(x_data, x_prior)

    def _fit(self, x_data, x_prior=None):
        """
        x_data  (N, D) integer codes of the training rows (x0).
        x_prior optional (N, D) prior sample (x1); by default N uniform draws.
        """
        self._fitted = False
        x0_real = self._check_codes(x_data)
        n_rows = x0_real.shape[0]
        if n_rows == 0:
            raise ValueError("cannot fit on an empty training set")
        if x_prior is None:
            x1_real = self.reference.sample_prior(n_rows, generator=self._generator(self.cfg.seed + 999))
        else:
            x1_real = self._check_codes(x_prior)
            if x1_real.shape[0] != n_rows:
                raise ValueError("x_prior must have as many rows as x_data")

        self.stage_log = []
        self.reference_trace = [dict(outer=0, mixing_rate=float(self.reference.mixing_rate))]

        for l in range(1, self.cfg.num_outer_iterations + 1):
            print(f"\n{'=' * 10} CSBM Outer Iteration L={l} {'=' * 10}")
            base = self.cfg.seed + 10000 * l

            if self.annealed and l % self.cfg.anneal_every == 0:
                rate = self.reference.mixing_rate * self.cfg.anneal_multiplier
                if self.cfg.min_mixing_rate is not None:
                    rate = max(rate, self.cfg.min_mixing_rate)
                self.reference.set_kernel(mixing_rate=rate)  # rebuilds every cached transition
                self.reference_trace.append(dict(outer=l, mixing_rate=float(rate)))

            # --- Forward update stage ---
            if l == 1:
                perm = torch.randperm(n_rows, generator=self._generator(base + 1), device=self.device)
                curr_x0, curr_x1, source = x0_real[perm], x1_real, "independent"
            else:
                print("Sampling new coupling (x0, x1) using backward model...")
                curr_x1 = x1_real
                curr_x0, _ = self.sampler.simulate(x_init=curr_x1, model=self.updater.backward_model,
                                                   direction="backward", seed=base + 2)
                source = "backward_model"
            n_upd, last = self._train_direction("forward", curr_x0, curr_x1, base + 3, f"L={l} | Forward Training")
            self.stage_log.append(dict(outer=l, direction="forward", coupling_source=source,
                                       anchored_endpoint="both" if l == 1 else "prior",
                                       n_updates=n_upd, last_epoch_loss=float(last)))

            # --- Backward update stage ---
            print("Sampling new coupling (x0, x1) using forward model...")
            new_x0 = x0_real
            new_x1, _ = self.sampler.simulate(x_init=new_x0, model=self.updater.forward_model,
                                              direction="forward", seed=base + 4)
            n_upd, last = self._train_direction("backward", new_x0, new_x1, base + 5, f"L={l} | Backward Training")
            self.stage_log.append(dict(outer=l, direction="backward", coupling_source="forward_model",
                                       anchored_endpoint="data", n_updates=n_upd, last_epoch_loss=float(last)))

        self._fitted = True
        return self

    @property
    def n_updates(self) -> int:
        return int(self.updater.n_forward_updates + self.updater.n_backward_updates)

    # ------------------------------------------------------------------ public sampling adapter
    @torch.no_grad()
    def sample(self, n_samples: int, seed: Optional[int] = None, batch_size: Optional[int] = None) -> torch.Tensor:
        """
        Exactly ``n_samples`` rows of integer codes, (n_samples, D), drawn by the
        backward network from the uniform prior. Every code lies inside its
        column's support: padded categories carry zero probability.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before sample().")
        if n_samples <= 0:
            raise ValueError("n_samples must be positive")
        gen = self._generator(seed if seed is not None else int(torch.seed()) % (2 ** 31))
        bs = int(batch_size) if batch_size else int(n_samples)
        if bs < 1:
            raise ValueError("batch_size must be positive")
        outs, done = [], 0
        while done < n_samples:
            m = min(bs, n_samples - done)
            x1 = self.reference.sample_prior(m, generator=gen)
            x0, _ = self.sampler.simulate(x_init=x1, model=self.updater.backward_model,
                                          direction="backward", generator=gen)
            outs.append(x0)
            done += m
        out = torch.cat(outs, dim=0)
        assert out.shape == (n_samples, len(self.cardinalities))
        return out

    # ------------------------------------------------------------------ checkpoint
    def state_dict(self) -> dict:
        return {
            "format": CHECKPOINT_FORMAT,
            "variant": self.variant,
            "config": asdict(self.cfg),
            "cardinalities": list(self.cardinalities),
            "is_ordered": [bool(b) for b in self.is_ordered.tolist()],
            "orientation": {"x0": "data", "x1": "prior", "generation": "backward"},
            "reference": self.reference.describe(),
            "reference_trace": self.reference_trace,
            "forward_model": {k: v.detach().cpu() for k, v in self.updater.forward_model.state_dict().items()},
            "backward_model": {k: v.detach().cpu() for k, v in self.updater.backward_model.state_dict().items()},
            "stage_log": self.stage_log,
            "fitted": self._fitted,
        }

    def save_checkpoint(self, path) -> None:
        torch.save(self.state_dict(), path)

    @classmethod
    def load_checkpoint(cls, path, device: Optional[str] = None) -> "CSBMSolver":
        state = torch.load(path, map_location="cpu", weights_only=False)
        if state.get("format") != CHECKPOINT_FORMAT:
            raise ValueError(f"unsupported CSBM checkpoint format: {state.get('format')!r}")
        cfg_cls = AnnealedCSBMConfig if state["variant"] == "csbm_annealed" else CSBMConfig
        cfg_dict = dict(state["config"])
        if device is not None:
            cfg_dict["device"] = device
        solver = cls(state["cardinalities"], torch.tensor(state["is_ordered"], dtype=torch.bool), cfg_cls(**cfg_dict))
        # An annealed fit ends on a different reference than it was configured with.
        solver.reference.set_kernel(mixing_rate=state["reference"]["mixing_rate"],
                                    ordered_bandwidth=state["reference"]["ordered_bandwidth"])
        solver.updater.forward_model.load_state_dict(state["forward_model"])
        solver.updater.backward_model.load_state_dict(state["backward_model"])
        solver.stage_log = state["stage_log"]
        solver.updater.n_forward_updates = sum(s["n_updates"] for s in solver.stage_log if s["direction"] == "forward")
        solver.updater.n_backward_updates = sum(s["n_updates"] for s in solver.stage_log if s["direction"] == "backward")
        solver.reference_trace = state["reference_trace"]
        solver._fitted = bool(state["fitted"])
        return solver
