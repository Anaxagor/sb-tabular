from dataclasses import asdict
from typing import List, Optional
import torch
from tqdm import tqdm
from sbtab.bridge.pathsampler import MixedPathSampler
from sbtab.bridge.reference import CategoricalReference, GaussianReference
from sbtab.bridge.sde import EulerMaruyama
from sbtab.bridge.timegrid import TimeGrid
from sbtab.models.neural.MixedMLP import MixedSbmMlp
from sbtab.solvers.msbm.config import MixedSBMConfig
from sbtab.solvers.msbm.updater import MixedSBMUpdater

CHECKPOINT_FORMAT = "sbtab.mixedsbm/2"


class MixedSBMSolver:
    """
    Mixed (numerical + categorical) Schrödinger bridge matching by IMF.

    Orientation (used consistently by training, couplings and generation):
      x0 = data  at t = 0,   x1 = prior at t = 1
      prior = N(0, I) on the numerical block, uniform on every categorical column
      'f' network: data -> prior drift / x1-logits
      'b' network: prior -> data drift / x0-logits;  generation runs 'b' from n = N to 0

    One unit-horizon TimeGrid is shared by the Brownian bridge, the categorical
    reference, the network clock and the sampler.

    IMF policy
      stage 0            independent coupling  data (x) prior   (declared initialiser)
      stage k >= 1       coupling re-simulated by the latest OPPOSITE-direction
                         network, anchored on its real start marginal: after 'f' the
                         pairs are (real data, simulated x1); after 'b' they are
                         (simulated x0, fixed prior sample).
      networks           one per direction; warm-started across stages unless
                         cfg.warm_start is False. Couplings are simulated under
                         no_grad, so no gradient crosses stages.
      snapshot           sample() uses the LAST 'b' stage.

    Either block may be absent (continuous_dim = 0 or cardinalities = []).
    """

    def __init__(self, continuous_dim: int, cardinalities: List[int],
                 is_ordered: torch.Tensor, cfg: "MixedSBMConfig"):
        self.cont_dim = int(continuous_dim)
        self.cardinalities = [int(c) for c in cardinalities]
        self.is_ordered = torch.as_tensor(is_ordered, dtype=torch.bool).view(-1)
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        if self.cont_dim == 0 and len(self.cardinalities) == 0:
            raise ValueError("MixedSBM needs at least one numerical or categorical column")
        self._validate_sequence(cfg.fb_sequence)
        torch.manual_seed(cfg.seed)

        self.timegrid = TimeGrid.uniform(cfg.num_steps, horizon=1.0)
        self.ref_gauss = GaussianReference(dim=self.cont_dim, device=self.device)
        self.ref_cat = None
        if len(self.cardinalities) > 0:
            self.ref_cat = CategoricalReference(
                cardinalities=self.cardinalities,
                is_ordered=self.is_ordered,
                timegrid=self.timegrid,
                mixing_rate=cfg.cat_mixing_rate,
                ordered_bandwidth=cfg.cat_ordered_bandwidth,
                device=self.device,
            )
        # Dynamics noise is always on: the drift is trained for the stochastic
        # bridge, and dropping the noise is not a marginal-preserving sampler.
        self.integrator = EulerMaruyama(noise=True, sigma=cfg.sigma)
        self.sampler = MixedPathSampler(
            timegrid=self.timegrid,
            reference=self.ref_cat,
            integrator=self.integrator,
        )
        self.models = {"f": self._new_model(), "b": self._new_model()}
        self.updaters = {d: MixedSBMUpdater(m, self.ref_cat, cfg, self.timegrid) for d, m in self.models.items()}
        self.snapshots = []
        self.stage_log = []
        self._fitted = False

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _validate_sequence(seq) -> None:
        if len(seq) == 0:
            raise ValueError("fb_sequence must not be empty")
        for d in seq:
            if d not in ("f", "b"):
                raise ValueError(f"Unknown direction in fb_sequence: {d}")
        for a, b in zip(seq[:-1], seq[1:]):
            if a == b:
                raise ValueError("fb_sequence must alternate: each stage is trained on the coupling "
                                 "of the opposite direction")

    def _new_model(self) -> MixedSbmMlp:
        return MixedSbmMlp(
            continuous_dim=self.cont_dim,
            cardinalities=self.cardinalities,
            cat_emb_dim=self.cfg.cat_emb_dim,
            hidden_dim=self.cfg.hidden_dim,
            time_dim=self.cfg.time_dim,
            n_layers=self.cfg.n_layers,
            dropout=self.cfg.dropout,
        ).to(self.device)

    def _sample_prior(self, n: int, generator: torch.Generator):
        if self.cont_dim > 0:
            prior_num = self.ref_gauss.sample(n, generator=generator)
        else:
            prior_num = torch.zeros((n, 0), device=self.device)
        if self.ref_cat is not None:
            prior_cat = self.ref_cat.sample_prior(n, generator=generator)
        else:
            prior_cat = torch.zeros((n, 0), dtype=torch.long, device=self.device)
        return prior_num, prior_cat

    def _generator(self, seed: int) -> torch.Generator:
        g = torch.Generator(device=str(self.device))
        g.manual_seed(int(seed))
        return g

    def _prepare(self, x_num, x_cat, n_rows: Optional[int] = None):
        if x_num is None:
            if n_rows is None:
                n_rows = x_cat.shape[0]
            x_num = torch.zeros((n_rows, 0))
        x_num = torch.as_tensor(x_num, dtype=torch.float32).to(self.device)
        if x_cat is None:
            x_cat = torch.zeros((x_num.shape[0], 0), dtype=torch.long)
        x_cat = torch.as_tensor(x_cat, device=self.device)
        if not torch.isfinite(x_cat).all() or (x_cat.is_floating_point() and not torch.equal(x_cat, x_cat.trunc())):
            raise ValueError("categorical codes must be finite integers")
        x_cat = x_cat.long()
        if x_num.ndim != 2 or x_cat.ndim != 2 or not torch.isfinite(x_num).all():
            raise ValueError("training blocks must be finite two-dimensional arrays")
        if x_num.shape[1] != self.cont_dim or x_cat.shape[1] != len(self.cardinalities):
            raise ValueError("training blocks do not match continuous_dim / cardinalities")
        if x_num.shape[0] != x_cat.shape[0]:
            raise ValueError("numerical and categorical blocks must have the same number of rows")
        for d, c in enumerate(self.cardinalities):
            if x_cat.shape[0] and (int(x_cat[:, d].min()) < 0 or int(x_cat[:, d].max()) >= c):
                raise ValueError(f"categorical column {d} has codes outside [0, {c})")
        return x_num, x_cat

    @torch.no_grad()
    def _generate_coupling(self, data_num, data_cat, prior_num, prior_cat, prev_dir, seed):
        if prev_dir is None:
            return data_num, data_cat, prior_num, prior_cat, "independent"

        if prev_dir == 'f':
            start_num, start_cat = data_num, data_cat
            direction = "forward"
        else:
            start_num, start_cat = prior_num, prior_cat
            direction = "backward"

        end_num, end_cat, _ = self.sampler.simulate(
            start_num, start_cat,
            model=self.models[prev_dir],
            direction=direction,
            seed=seed,
        )

        if prev_dir == 'f':
            return start_num, start_cat, end_num, end_cat, "forward_model"
        else:
            return end_num, end_cat, start_num, start_cat, "backward_model"

    # ------------------------------------------------------------------ training
    def fit(self, train_num, train_cat=None):
        """Seed dropout and warm-start reinitialisation independently of the caller's RNG."""
        self._fitted = False
        devices = [self.device] if self.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(self.cfg.seed + 104729)
            return self._fit(train_num, train_cat)

    def _fit(self, train_num, train_cat=None):
        """
        Train the model according to the sequence of directions in cfg.fb_sequence.
        """
        train_num, train_cat = self._prepare(train_num, train_cat)
        N = train_num.shape[0]
        if N == 0:
            raise ValueError("cannot fit on an empty training set")
        seeds = {"prior": self.cfg.seed + 999}
        prior_num, prior_cat = self._sample_prior(N, self._generator(seeds["prior"]))

        prev_dir = None
        self.snapshots = []
        self.stage_log = []

        total_stages = len(self.cfg.fb_sequence)
        outer_pbar = tqdm(total=total_stages, desc="MSBM iterations", unit="stage")

        for idx, direction_short in enumerate(self.cfg.fb_sequence):
            direction_full = 'forward' if direction_short == 'f' else 'backward'
            outer_pbar.set_postfix(stage=f"{direction_full} {idx+1}/{total_stages}")

            coupling_seed = self.cfg.seed + 10000 + idx
            train_seed = self.cfg.seed + 20000 + idx
            z0_num, z0_cat, z1_num, z1_cat, source = self._generate_coupling(
                train_num, train_cat, prior_num, prior_cat, prev_dir, seed=coupling_seed
            )

            if not self.cfg.warm_start and any(s["fb"] == direction_short for s in self.snapshots):
                self.models[direction_short] = self._new_model()
                self.updaters[direction_short] = MixedSBMUpdater(
                    self.models[direction_short], self.ref_cat, self.cfg, self.timegrid)

            n_updates, last_loss = self.updaters[direction_short].train_epochs(
                direction_short, z0_num, z0_cat, z1_num, z1_cat,
                epochs=self.cfg.epochs_per_direction, seed=train_seed,
            )

            model = self.models[direction_short]
            snap_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            self.snapshots.append({"fb": direction_short, "stage": idx, "state": snap_state})
            self.stage_log.append({
                "stage": idx,
                "fb": direction_short,
                "coupling_source": source,
                # which endpoint of the coupling consists of real (non-simulated) rows
                "anchored_endpoint": {"independent": "both", "forward_model": "data", "backward_model": "prior"}[source],
                "n_updates": int(n_updates),
                "last_epoch_loss": float(last_loss),
                "coupling_seed": None if prev_dir is None else int(coupling_seed),
                "train_seed": int(train_seed),
            })

            prev_dir = direction_short

            outer_pbar.update(1)

        outer_pbar.close()
        self.seeds = seeds
        self._fitted = True
        return self

    @property
    def n_updates(self) -> int:
        return int(sum(s["n_updates"] for s in self.stage_log))

    def generation_stage(self) -> int:
        """Index of the snapshot used by sample(): the last 'b' stage."""
        for item in reversed(self.snapshots):
            if item["fb"] == "b":
                return int(item["stage"])
        raise RuntimeError("No backward snapshot found. Ensure fb_sequence contains at least one 'b'.")

    # ------------------------------------------------------------------ sampling
    @torch.no_grad()
    def sample(self, n_samples, seed=None, batch_size: Optional[int] = None):
        if not self._fitted:
            raise RuntimeError("Call fit() before sample().")
        if n_samples <= 0:
            raise ValueError("n_samples must be positive")

        stage = self.generation_stage()
        b_state = next(s["state"] for s in self.snapshots if s["stage"] == stage)
        devices = [self.device] if self.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            tmp_model = self._new_model()
        tmp_model.load_state_dict(b_state)
        tmp_model.eval()

        # One generator drives the prior draw and all dynamics noise of every
        # batch, so batches are not reseeded to identical outputs.
        gen = None if seed is None else self._generator(seed)
        if gen is None:
            gen = self._generator(int(torch.seed()) % (2 ** 31))

        bs = int(batch_size) if batch_size else int(n_samples)
        if bs < 1:
            raise ValueError("batch_size must be positive")
        outs_num, outs_cat = [], []
        done = 0
        while done < n_samples:
            m = min(bs, n_samples - done)
            start_num, start_cat = self._sample_prior(m, gen)
            gen_num, gen_cat, _ = self.sampler.simulate(
                start_num, start_cat,
                model=tmp_model,
                direction="backward",
                generator=gen,
            )
            outs_num.append(gen_num)
            outs_cat.append(gen_cat)
            done += m
        return torch.cat(outs_num, dim=0), torch.cat(outs_cat, dim=0)

    # ------------------------------------------------------------------ checkpoint
    def state_dict(self) -> dict:
        """Inference-complete state; reload never refits."""
        return {
            "format": CHECKPOINT_FORMAT,
            "config": asdict(self.cfg),
            "continuous_dim": self.cont_dim,
            "cardinalities": list(self.cardinalities),
            "is_ordered": [bool(b) for b in self.is_ordered.tolist()],
            "orientation": {"x0": "data", "x1": "prior", "generation": "backward"},
            "timegrid": [float(v) for v in self.timegrid._grid64().tolist()],
            "reference": None if self.ref_cat is None else self.ref_cat.describe(),
            "snapshots": self.snapshots,
            "stage_log": self.stage_log,
            "seeds": getattr(self, "seeds", {}),
            "fitted": self._fitted,
        }

    def save_checkpoint(self, path) -> None:
        torch.save(self.state_dict(), path)

    @classmethod
    def load_checkpoint(cls, path, device: Optional[str] = None) -> "MixedSBMSolver":
        state = torch.load(path, map_location="cpu", weights_only=False)
        if state.get("format") != CHECKPOINT_FORMAT:
            raise ValueError(f"unsupported MixedSBM checkpoint format: {state.get('format')!r}")
        cfg_dict = dict(state["config"])
        cfg_dict["fb_sequence"] = tuple(cfg_dict["fb_sequence"])
        if device is not None:
            cfg_dict["device"] = device
        solver = cls(state["continuous_dim"], state["cardinalities"],
                     torch.tensor(state["is_ordered"], dtype=torch.bool), MixedSBMConfig(**cfg_dict))
        solver.snapshots = state["snapshots"]
        solver.stage_log = state["stage_log"]
        solver.seeds = state.get("seeds", {})
        solver._fitted = bool(state["fitted"])
        for d in ("f", "b"):
            last = [s for s in solver.snapshots if s["fb"] == d]
            if last:
                solver.models[d].load_state_dict(last[-1]["state"])
        return solver
