from typing import Optional

import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from sbtab.bridge.losses import MixedSBMLoss
from sbtab.bridge.timegrid import TimeGrid

class MixedSBMUpdater:
    """
    Bridge-matching updates for one direction's network.

    Unit-horizon Brownian bridge between x0 (t = 0) and x1 (t = 1):

        X_t = (1 - t) x0 + t x1 + sigma sqrt(t (1 - t)) eps

    forward  target (drift in t,      units x / time): (x1 - X_t) / (1 - t) = (x1 - x0) - sigma sqrt(t / (1 - t)) eps
    backward target (drift in s=1-t,  units x / time): (x0 - X_t) / t       = -(x1 - x0) - sigma sqrt((1 - t) / t) eps

    The state index n is drawn per direction so that every index the sampler
    visits is trained: forward n in [0, N-1] (t < 1), backward n in [1, N]
    (t > 0). Both targets are finite on those ranges without any clamping, and
    the numerical block, the categorical block and the network all use the same
    n and the same t = grid[n].
    """

    def __init__(self, model, ref_cat, cfg: "MixedSBMConfig", timegrid: TimeGrid):
        self.model = model
        self.ref_cat = ref_cat
        self.cfg = cfg
        self.timegrid = timegrid
        if abs(timegrid.T - 1.0) > 1e-12:
            raise ValueError("MixedSBM uses a unit-time Brownian bridge; the grid horizon must be 1")
        self.grid = timegrid.grid().to(torch.device(cfg.device))
        self.N = timegrid.num_steps
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
        self.loss_fn = MixedSBMLoss(reference=ref_cat, lambda_num=cfg.lambda_num, lambda_cat=cfg.lambda_cat,
                                    ce_lambda=cfg.ce_lambda)
        self.n_updates = 0

    def _make_training_tuple(self, z0_num, z0_cat, z1_num, z1_cat, direction,
                             generator: Optional[torch.Generator] = None):
        B = z0_num.shape[0]
        device = z0_num.device
        if direction == 'f':
            n = torch.randint(0, self.N, (B,), device=device, generator=generator)
        elif direction == 'b':
            n = torch.randint(1, self.N + 1, (B,), device=device, generator=generator)
        else:
            raise ValueError(f"Unknown direction: {direction}")
        t = self.grid[n].view(-1, 1)

        x_t_num, target_num = z0_num, None
        if z0_num.shape[1] > 0:
            noise_num = torch.randn(z0_num.shape, device=device, dtype=z0_num.dtype, generator=generator)
            x_t_num = (1 - t) * z0_num + t * z1_num
            x_t_num = x_t_num + self.cfg.sigma * torch.sqrt(t * (1 - t)) * noise_num

            delta_num = z1_num - z0_num
            if direction == 'f':
                target_num = delta_num - self.cfg.sigma * torch.sqrt(t / (1 - t)) * noise_num
            else:
                target_num = -delta_num - self.cfg.sigma * torch.sqrt((1 - t) / t) * noise_num

        x_t_cat = z0_cat
        if z0_cat.shape[1] > 0:
            x_t_cat = self.ref_cat.sample_x_t(z0_cat, z1_cat, n, generator=generator)
        target_cat = z1_cat if direction == 'f' else z0_cat

        return x_t_num, x_t_cat, t, n, target_num, target_cat

    def train_step(self, z0_num, z0_cat, z1_num, z1_cat, direction, generator=None):
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        x_t_num, x_t_cat, t, n, target_num, target_cat = self._make_training_tuple(
            z0_num, z0_cat, z1_num, z1_cat, direction, generator=generator
        )
        pred_num, pred_logits_cat = self.model(x_t_num, x_t_cat, t)

        dir_str = "forward" if direction == 'f' else "backward"

        loss = self.loss_fn(
            pred_num=pred_num,
            target_num=target_num,
            pred_logits_cat=pred_logits_cat,
            true_cat=target_cat,
            x_t_cat=x_t_cat,
            n=n,
            K=self.N,
            direction=dir_str,
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite MixedSBM loss in direction {direction!r}")
        loss.backward()
        if self.cfg.grad_clip:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
        self.optimizer.step()
        self.n_updates += 1
        return loss.item()

    def train_epochs(self, direction, z0_num, z0_cat, z1_num, z1_cat, epochs, seed: Optional[int] = None):
        """Returns (number of optimizer updates, mean loss of the last epoch)."""
        dataset = TensorDataset(z0_num, z0_cat, z1_num, z1_cat)
        # CPU generator for the shuffle; a device generator for bridge/noise draws.
        shuffle_gen = torch.Generator()
        noise_gen = torch.Generator(device=str(z0_num.device))
        if seed is not None:
            shuffle_gen.manual_seed(int(seed))
            noise_gen.manual_seed(int(seed) + 1)
        # Partial batches are kept: with drop_last=True a dataset smaller than
        # batch_size produced zero batches and the stage silently trained nothing.
        # The MLP has no batch-statistics layers, so a batch of one row is valid.
        loader = DataLoader(dataset, batch_size=self.cfg.batch_size, shuffle=True, drop_last=False,
                            generator=shuffle_gen)

        updates_before = self.n_updates
        avg_loss = float("nan")
        pbar = tqdm(range(epochs), desc=f"Training {direction}-direction", unit="epoch")
        for _ in pbar:
            total_loss = 0.0
            n_batches = 0
            for b0n, b0c, b1n, b1c in loader:
                loss = self.train_step(b0n, b0c, b1n, b1c, direction, generator=noise_gen)
                total_loss += loss
                n_batches += 1
            avg_loss = total_loss / n_batches
            pbar.set_postfix(avg_loss=f"{avg_loss:.4f}")

        n_new = self.n_updates - updates_before
        if epochs > 0 and n_new <= 0:
            raise RuntimeError("MixedSBM stage performed no optimizer updates")
        return n_new, avg_loss
