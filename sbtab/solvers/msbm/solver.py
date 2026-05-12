from typing import List
import torch
from tqdm import tqdm
from sbtab.bridge.pathsampler import MixedPathSampler
from sbtab.bridge.reference import CategoricalReference, GaussianReference
from sbtab.bridge.sde import EulerMaruyama
from sbtab.bridge.timegrid import TimeGrid
from sbtab.models.neural.MixedMLP import MixedSbmMlp
from sbtab.solvers.msbm import MixedSBMUpdater


class MixedSBMSolver:
    def __init__(self, continuous_dim: int, cardinalities: List[int],
                 is_ordered: torch.Tensor, cfg: "MixedSBMConfig"):
        self.cont_dim = continuous_dim
        self.cardinalities = cardinalities
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        torch.manual_seed(cfg.seed)

        self.ref_gauss = GaussianReference(dim=continuous_dim, device=self.device)
        self.ref_cat = CategoricalReference(
            cardinalities=cardinalities,
            is_ordered=is_ordered,
            total_number_of_q_powers=cfg.num_steps,
            alpha=cfg.alpha,
            device=self.device,
        )
        self.integrator = EulerMaruyama(noise=True, sigma=cfg.sigma)
        self.timegrid = TimeGrid(num_steps=cfg.num_steps)
        self.sampler = MixedPathSampler(
            timegrid=self.timegrid,
            reference=self.ref_cat,
            integrator=self.integrator,
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
        self.updater = MixedSBMUpdater(self.model, self.ref_cat, cfg)
        self.snapshots = []
        self._fitted = False

    @torch.no_grad()
    def _generate_coupling(self, data_num, data_cat, prior_num, prior_cat,
                           prev_state, prev_dir, seed):
        if prev_state is None:
            return data_num, data_cat, prior_num, prior_cat

        tmp_model = MixedSbmMlp(
            continuous_dim=self.cont_dim,
            cardinalities=self.cardinalities,
            cat_emb_dim=self.cfg.cat_emb_dim,
            hidden_dim=self.cfg.hidden_dim,
            time_dim=self.cfg.time_dim,
            n_layers=self.cfg.n_layers,
            dropout=self.cfg.dropout,
        ).to(self.device)
        tmp_model.load_state_dict(prev_state)
        tmp_model.eval()

        if prev_dir == 'f':
            start_num, start_cat = data_num, data_cat
            direction = "forward"
        else:
            start_num, start_cat = prior_num, prior_cat
            direction = "backward"

        end_num, end_cat, _ = self.sampler.simulate(
            start_num, start_cat,
            model=tmp_model,
            direction=direction,
            seed=seed,
        )

        if prev_dir == 'f':
            return start_num, start_cat, end_num, end_cat
        else:
            return end_num, end_cat, start_num, start_cat

    def _train_direction(self, direction, z0_num, z0_cat, z1_num, z1_cat):
        self.updater.train_epochs(direction, z0_num, z0_cat, z1_num, z1_cat,
                                  epochs=self.cfg.epochs_per_direction)

    def fit(self, train_num, train_cat):
        """
        Train the model according to the sequence of directions in cfg.fb_sequence.
        """
        N = train_num.shape[0]
        prior_num = self.ref_gauss.sample(N, seed=self.cfg.seed + 999).to(self.device)
        prior_cat = torch.stack([
            torch.randint(0, c, (N,), device=self.device) for c in self.cardinalities
        ], dim=1)

        prev_state = None
        prev_dir = None

        total_stages = len(self.cfg.fb_sequence)
        outer_pbar = tqdm(total=total_stages, desc="MSBM iterations", unit="stage")

        for idx, direction_short in enumerate(self.cfg.fb_sequence):
            if direction_short == 'f':
                direction_full = 'forward'
            elif direction_short == 'b':
                direction_full = 'backward'
            else:
                raise ValueError(f"Unknown direction in fb_sequence: {direction_short}")

            outer_pbar.set_postfix(stage=f"{direction_full} {idx+1}/{total_stages}")

            z0_num, z0_cat, z1_num, z1_cat = self._generate_coupling(
                train_num, train_cat, prior_num, prior_cat,
                prev_state, prev_dir,
                seed=self.cfg.seed + 10000 + idx
            )

            self._train_direction(direction_short, z0_num, z0_cat, z1_num, z1_cat)

            snap_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
            self.snapshots.append({"fb": direction_short, "state": snap_state})

            prev_state = snap_state
            prev_dir = direction_short

            outer_pbar.update(1)

        outer_pbar.close()
        self._fitted = True
        return self

    @torch.no_grad()
    def sample(self, n_samples, seed=None):
        if not self._fitted:
            raise RuntimeError("Call fit() before sample().")

        b_state = None
        for item in reversed(self.snapshots):
            if item["fb"] == "b":
                b_state = item["state"]
                break
        if b_state is None:
            raise RuntimeError("No backward snapshot found. Ensure fb_sequence contains at least one 'b'.")

        tmp_model = MixedSbmMlp(
            continuous_dim=self.cont_dim,
            cardinalities=self.cardinalities,
            cat_emb_dim=self.cfg.cat_emb_dim,
            hidden_dim=self.cfg.hidden_dim,
            time_dim=self.cfg.time_dim,
            n_layers=self.cfg.n_layers,
            dropout=self.cfg.dropout,
        ).to(self.device)
        tmp_model.load_state_dict(b_state)
        tmp_model.eval()

        start_num = self.ref_gauss.sample(n_samples, seed=seed).to(self.device)
        start_cat = torch.stack([
            torch.randint(0, c, (n_samples,), device=self.device) for c in self.cardinalities
        ], dim=1)

        gen_num, gen_cat, _ = self.sampler.simulate(
            start_num, start_cat,
            model=tmp_model,
            direction="backward",
            seed=seed,
        )
        return gen_num, gen_cat
