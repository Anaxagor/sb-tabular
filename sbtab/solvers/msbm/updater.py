from typing import Literal, Tuple

import torch
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
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.loss_fn = MixedSBMLoss(reference=ref_cat, lambda_num=cfg.lambda_num, lambda_cat=cfg.lambda_cat, ce_lambda=cfg.ce_lambda)

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
            t: Time tensor scaled to [0, 1] for model input.
            n: Random integer time steps: [0, K-1] for 'f', [1, K] for 'b'
            (ranges match the sampling loop in MixedPathSampler).
            target_num: Target drift/velocity vector for continuous features regression.
            target_cat: Ground-truth target indices (z1_cat for forward, z0_cat for backward).
        """
        B = z0_num.shape[0] if self.has_cont else z0_cat.shape[0]
        device = z0_num.device if self.has_cont else z0_cat.device

        if direction == 'f':
            n = torch.randint(0, self.cfg.num_steps, (B,), device=device)
        else:
            n = torch.randint(1, self.cfg.num_steps + 1, (B,), device=device)
        t = n.float().view(-1, 1) / self.cfg.num_steps

        if self.has_cont:
            noise_num = torch.randn_like(z0_num)
            x_t_num = (1 - t) * z0_num + t * z1_num
            x_t_num = x_t_num + self.cfg.sigma * torch.sqrt(t * (1 - t)) * noise_num
            delta_num = z1_num - z0_num
            if direction == 'f':
                target_num = delta_num - self.cfg.sigma * torch.sqrt(t / (1 - t + 1e-12)) * noise_num
            else:
                target_num = -delta_num - self.cfg.sigma * torch.sqrt((1 - t) / (t + 1e-12)) * noise_num
        else:
            x_t_num, target_num = z0_num, z1_num

        if self.has_cat:
            x_t_cat = self.ref_cat.sample_x_t(z0_cat, z1_cat, n, self.cfg.num_steps)
            target_cat = z1_cat if direction == 'f' else z0_cat
        else:
            x_t_cat, target_cat = z0_cat, z1_cat

        return x_t_num, x_t_cat, t, n, target_num, target_cat

    def train_step(
        self,
        z0_num: torch.Tensor | None,
        z0_cat: torch.Tensor | None,
        z1_num: torch.Tensor | None,
        z1_cat: torch.Tensor | None,
        direction: Literal['f', 'b']
    ) -> torch.Tensor:
        """
        One training step of the MLP backbone model.

        Args:
            z0_num (torch.Tensor | None): Initial continuous features' distribution (z_0). None if no continuous features.
            z0_cat (torch.Tensor | None): Initial categorical features' indices (z_0). None if no categorical features.
            z1_num (torch.Tensor | None): Target continuous features' distribution (z_1). None if no continuous features.
            z1_cat (torch.Tensor | None): Target categorical features' indices (z_1). None if no categorical features.
            direction (Literal['f', 'b']): Direction of the imf loop.

        Returns:
            torch.Tensor: Training loss.
        """
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
        return loss.detach()

    def train_epochs(
        self,
        direction: Literal['f', 'b'],
        z0_num: torch.Tensor | None,
        z0_cat: torch.Tensor | None,
        z1_num: torch.Tensor | None,
        z1_cat: torch.Tensor | None,
        epochs: int | None = None,
        steps: int | None = None,
        min_steps: int | None = None
    ) -> None:
        """
        Trains MLP model in one chosen direction for a resolved step budget.

        Budget resolution (first match wins):
            1. explicit `steps` argument — fixed number of optimizer steps;
            2. explicit `epochs` argument — epochs * ceil(N / batch_size) steps;
            3. cfg.steps_per_direction;
            4. cfg.epochs_per_direction * ceil(N / batch_size);
            5. otherwise raises ValueError.
        The resolved budget is then lifted to `min_steps` (floor in steps; None = no floor).

        Batching: on-device index slicing, a fresh permutation every ceil(N / batch_size)
        steps (equivalent to DataLoader(shuffle=True, drop_last=False)); the loss is synced
        to the host only once per log_every steps.

        Args:
            direction: Direction of the imf loop ('f' or 'b').
            z0_num: Continuous features at the start of the bridge (Shape: [N, cont_dim]).
                None if no continuous features.
            z0_cat: Categorical feature indices at the start of the bridge (Shape: [N, cat_dim]).
                None if no categorical features.
            z1_num: Continuous features at the end of the bridge. None if no continuous features.
            z1_cat: Categorical feature indices at the end of the bridge. None if no categorical features.
            epochs: Explicit epoch budget. Overrides cfg; lower priority than `steps`.
            steps: Explicit step budget. Highest priority.
            min_steps: Floor on the resolved budget, in steps. None = no floor applied.

        Returns:
            None
        """
        data = z0_num if self.has_cont else z0_cat
        n_train = int(data.shape[0])
        if n_train == 0:
            return
        device = data.device
        batch_size = self.cfg.batch_size
        batches_per_epoch = max(1, (n_train + batch_size - 1) // batch_size)

        if steps is not None:
            total_steps = int(steps)
        elif epochs is not None:
            total_steps = int(epochs) * batches_per_epoch
        elif self.cfg.steps_per_direction is not None:
            total_steps = self.cfg.steps_per_direction
        elif self.cfg.epochs_per_direction is not None:
            total_steps = self.cfg.epochs_per_direction * batches_per_epoch
        else:
            raise ValueError("No training budget: pass steps=/epochs= or set "
                             "cfg.steps_per_direction / cfg.epochs_per_direction")

        if min_steps is not None:
            total_steps = max(total_steps, int(min_steps))
        if total_steps <= 0:
            return

        self.model.train()
        log_every = max(1, total_steps // 25)
        loss_acc = torch.zeros((), device=device)
        steps_since_log = 0

        with tqdm(total=total_steps, desc=f"Training {direction}-direction", unit="step") as pbar:
            for step in range(total_steps):
                if step % batches_per_epoch == 0:
                    perm = torch.randperm(n_train, device=device)
                b = step % batches_per_epoch
                idx = perm[b * batch_size:(b + 1) * batch_size]

                loss = self.train_step(
                    None if z0_num is None else z0_num[idx],
                    None if z0_cat is None else z0_cat[idx],
                    None if z1_num is None else z1_num[idx],
                    None if z1_cat is None else z1_cat[idx],
                    direction,
                )
                loss_acc += loss
                steps_since_log += 1

                if (step + 1) % log_every == 0 or step == total_steps - 1:
                    avg = (loss_acc / steps_since_log).item()
                    pbar.set_postfix(avg_loss=f"{avg:.4f}")
                    loss_acc.zero_()
                    steps_since_log = 0
                pbar.update(1)
