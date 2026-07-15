from typing import Literal, Tuple

import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from sbtab.bridge.losses import MixedSBMLoss
from sbtab.bridge.reference import CategoricalReference

class MixedSBMUpdater:
    def __init__(
        self,
        model: "MixedSbmMlp",
        ref_cat: CategoricalReference,
        cfg: "MixedSBMConfig",
        has_cont: bool,
        has_cat: bool
    )-> None:
        """
        MSBM updater part initialization. Purpose: train one direction of imf loop.

        Args:
            model (MixedSbmMlp): MSBM backbone model.
            ref_cat (CategoricalReference): Categorical reference process.
            cfg (MixedSBMConfig): Configuration object.
            has_cont(bool): Flag that tells whether there are any continuous features.
            has_cat(bool): Flag that tells whether there are any categorical features.

        Returns:
            None
        """
        self.model = model
        self.ref_cat = ref_cat
        self.cfg = cfg
        self.has_cont = has_cont
        self.has_cat = has_cat
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
        self.loss_fn = MixedSBMLoss(reference=ref_cat, lambda_num=cfg.lambda_num, lambda_cat=cfg.lambda_cat)

    def _make_training_tuple(
        self,
        z0_num: torch.Tensor | None,
        z0_cat: torch.Tensor | None,
        z1_num: torch.Tensor | None,
        z1_cat: torch.Tensor | None,
        direction: Literal['f', 'b']
    ) -> Tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """
        Creates training tuple by sampling random time steps and interpolating data.

        Args:
            z0_num: Continuous features at the start of the bridge (Shape: [B, cont_dim]).
            z0_cat: Categorical feature indices at the start of the bridge (Shape: [B, cat_dim], dtype=long).
            z1_num: Continuous features at the end of the bridge (Shape: [B, cont_dim]).
            z1_cat: Categorical feature indices at the end of the bridge (Shape: [B, cat_dim], dtype=long).
            direction: Direction of the SBM loop ('f' for forward, 'b' for backward).

        Returns:
            x_t_num: Noisy interpolated continuous features at time t.
            x_t_cat: Categorical indices sampled from reference process at step n.
            t_safe: Clamped time tensor scaled to [0, 1] for model input.
            n: Random integer time steps sampled uniformly from [1, num_steps - 1].
            target_num: Target drift/velocity vector for continuous features regression.
            target_cat: Ground-truth target indices (z1_cat for forward, z0_cat for backward).
        """
        B = z0_num.shape[0] if self.has_cont else z0_cat.shape[0]
        device = z0_num.device if self.has_cont else z0_cat.device

        n = torch.randint(1, self.cfg.num_steps, (B,), device=device)
        t = n.float().view(-1, 1) / self.cfg.num_steps
        min_t = max(1.0 / self.cfg.num_steps, self.cfg.eps)
        t_safe = t.clamp(min=min_t, max=1.0 - min_t)

        if self.has_cont:
            noise_num = torch.randn_like(z0_num)
            x_t_num = (1 - t_safe) * z0_num + t_safe * z1_num
            x_t_num = x_t_num + self.cfg.sigma * torch.sqrt(t_safe * (1 - t_safe)) * noise_num
            delta_num = z1_num - z0_num
            if direction == 'f':
                target_num = delta_num - self.cfg.sigma * torch.sqrt(t_safe / (1 - t_safe + 1e-12)) * noise_num
            else:
                target_num = -delta_num - self.cfg.sigma * torch.sqrt((1 - t_safe) / (t_safe + 1e-12)) * noise_num
        else:
            x_t_num, target_num = z0_num, z1_num

        if self.has_cat:
            x_t_cat = self.ref_cat.sample_x_t(z0_cat, z1_cat, n, self.cfg.num_steps)
            target_cat = z1_cat if direction == 'f' else z0_cat
        else:
            x_t_cat, target_cat = z0_cat, z1_cat

        return x_t_num, x_t_cat, t_safe, n, target_num, target_cat

    def train_step(
        self,
        z0_num: torch.Tensor | None,
        z0_cat: torch.Tensor | None,
        z1_num: torch.Tensor | None,
        z1_cat: torch.Tensor | None,
        direction: Literal['f', 'b']
    ) -> float:
        """
        One training step of the MLP backbone model.

        Args:
            z0_num (torch.Tensor | None): Initial continuous features' distribution (z_0). None if no continuous features.
            z0_cat (torch.Tensor | None): Initial categorical features' indices (z_0). None if no categorical features.
            z1_num (torch.Tensor | None): Target continuous features' distribution (z_1). None if no continuous features.
            z1_cat (torch.Tensor | None): Target categorical features' indices (z_1). None if no categorical features.
            direction (Literal['f', 'b']): Direction of the imf loop.

        Returns:
            float: Training loss value.
        """
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        x_t_num, x_t_cat, t, n, target_num, target_cat = self._make_training_tuple(
            z0_num, z0_cat, z1_num, z1_cat, direction
        )
        pred_num, pred_logits_cat = self.model(x_t_num, x_t_cat, t)

        loss = self.loss_fn(
            pred_num=pred_num,
            true_num=target_num,
            pred_logits_cat=pred_logits_cat,
            true_cat=target_cat,
            x_t_cat=x_t_cat,
            n=n,
            K=self.cfg.num_steps,
            direction=direction,
        )
        loss.backward()
        if self.cfg.grad_clip:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
        self.optimizer.step()
        return loss.item()

    def train_epochs(
        self,
        direction: Literal['f', 'b'],
        z0_num: torch.Tensor | None,
        z0_cat: torch.Tensor | None,
        z1_num: torch.Tensor | None,
        z1_cat: torch.Tensor | None,
        epochs: int
    ) -> None:
        """
        Trains MLP model in one chosen direction for a specified number of epochs.

        Args:
            direction (Literal['f', 'b']): Direction of the imf loop.
            z0_num (torch.Tensor | None): Initial continuous features' distribution (z_0). None if no continuous features.
            z0_cat (torch.Tensor | None): Initial categorical features' indices (z_0). None if no categorical features.
            z1_num (torch.Tensor | None): Target continuous features' distribution (z_1). None if no continuous features.
            z1_cat (torch.Tensor | None): Target categorical features' indices (z_1). None if no categorical features.
            epochs (int): Number of epochs to train.

        Returns:
            None
        """
        dataset = TensorDataset(z0_num, z0_cat, z1_num, z1_cat)
        loader = DataLoader(dataset, batch_size=self.cfg.batch_size, shuffle=True)

        pbar = tqdm(range(epochs), desc=f"Training {direction}-direction", unit="epoch")
        for _ in pbar:
            total_loss = 0.0
            n_batches = 0
            for b0n, b0c, b1n, b1c in loader:
                loss = self.train_step(b0n, b0c, b1n, b1c, direction)
                total_loss += loss
                n_batches += 1
            avg_loss = total_loss / n_batches if n_batches > 0 else 0.0
            pbar.set_postfix(avg_loss=f"{avg_loss:.4f}")