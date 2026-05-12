import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from sbtab.bridge.losses import MixedSBMLoss

class MixedSBMUpdater:
    def __init__(self, model, ref_cat, cfg: "MixedSBMConfig"):
        self.model = model
        self.ref_cat = ref_cat
        self.cfg = cfg
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
        self.loss_fn = MixedSBMLoss(reference=ref_cat, lambda_num=cfg.lambda_num, lambda_cat=cfg.lambda_cat)

    def _make_training_tuple(self, z0_num, z0_cat, z1_num, z1_cat, direction):
        B = z0_num.shape[0]
        device = z0_num.device
        n = torch.randint(1, self.cfg.num_steps, (B,), device=device)
        t = n.float().view(-1, 1) / self.cfg.num_steps
        t_safe = t.clamp(min=0.01, max=0.99)

        noise_num = torch.randn_like(z0_num)
        x_t_num = (1 - t_safe) * z0_num + t_safe * z1_num
        x_t_num = x_t_num + self.cfg.sigma * torch.sqrt(t_safe * (1 - t_safe)) * noise_num

        x_t_cat = self.ref_cat.sample_x_t(z0_cat, z1_cat, n, self.cfg.num_steps)

        delta_num = z1_num - z0_num
        if direction == 'f':
            target_num = delta_num - self.cfg.sigma * torch.sqrt(t_safe / (1 - t_safe + 1e-12)) * noise_num
            target_cat = z1_cat
        else:
            target_num = -delta_num - self.cfg.sigma * torch.sqrt((1 - t_safe) / (t_safe + 1e-12)) * noise_num
            target_cat = z0_cat

        return x_t_num, x_t_cat, t_safe, n, target_num, target_cat

    def train_step(self, z0_num, z0_cat, z1_num, z1_cat, direction):
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        x_t_num, x_t_cat, t, n, target_num, target_cat = self._make_training_tuple(
            z0_num, z0_cat, z1_num, z1_cat, direction
        )
        pred_num, pred_logits_cat = self.model(x_t_num, x_t_cat, t)

        if torch.isnan(pred_num).any() or torch.isnan(pred_logits_cat).any():
            raise ValueError("Model output contains NaNs! Training is unstable.")

        if torch.isinf(pred_logits_cat).any():
            raise ValueError("Model output contains Infs!")

        dir_str = "forward" if direction == 'f' else "backward"

        loss = self.loss_fn(
            pred_num=pred_num,
            target_num=target_num,
            pred_logits_cat=pred_logits_cat,
            true_cat=target_cat,
            x_t_cat=x_t_cat,
            n=n,
            K=self.cfg.num_steps,
            direction=dir_str,
        )
        loss.backward()
        if self.cfg.grad_clip:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)

        is_nan = False
        for param in self.model.parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                is_nan = True
                break

        if is_nan:
            raise ValueError("Exploding gradients: NaN detected")

        self.optimizer.step()
        return loss.item()

    def train_epochs(self, direction, z0_num, z0_cat, z1_num, z1_cat, epochs):
        dataset = TensorDataset(z0_num, z0_cat, z1_num, z1_cat)
        loader = DataLoader(dataset, batch_size=self.cfg.batch_size, shuffle=True, drop_last=True)

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