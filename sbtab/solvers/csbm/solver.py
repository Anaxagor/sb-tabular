from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from sbtab.bridge.pathsampler import DiscretePathSampler
from sbtab.solvers.csbm import CSBMUpdater


@dataclass
class CSBMSolver:
    updater: "CSBMUpdater"
    sampler: "DiscretePathSampler"
    num_outer_iterations: int
    epochs: int = 15
    batch_size: int = 264

    def fit(self, dataloader_p1, dataloader_p0, scheduler_outer_iterations=5, scheduler_alpha_multiplier=0.9):
        K = self.sampler.timegrid.num_steps
        ref = self.sampler.reference
        device = ref.device

        all_x1_real = torch.cat([b[0] for b in dataloader_p1]).to(device)
        all_x0_real = torch.cat([b[0] for b in dataloader_p0]).to(device)

        for l in range(1, self.num_outer_iterations + 1):
            print(f"\n{'=' * 10} CSBM Outer Iteration L={l} {'=' * 10}")

            # Alpha decay
            if l % scheduler_outer_iterations == 0:
                ref.update_alpha(ref.alpha * scheduler_alpha_multiplier)

            # --- Forward update stage ---
            if l == 1:
                indices = torch.randperm(len(all_x0_real))
                curr_x0 = all_x0_real[indices[:len(all_x1_real)]]
                curr_x1 = all_x1_real
            else:
                print("Sampling new coupling (x1, x0) using backward model...")
                curr_x1 = all_x1_real
                curr_x0, _ = self.sampler.simulate(
                    x_init=curr_x1,
                    model=self.updater.backward_model,
                    direction="backward"
                )

            coupling_loader = DataLoader(TensorDataset(curr_x0, curr_x1), batch_size=self.batch_size, shuffle=True)
            for epoch in range(self.epochs):
                pbar = tqdm(coupling_loader, desc=f"L={l} | Forward Training")
                for x0_batch, x1_batch in pbar:
                    B = x1_batch.shape[0]
                    n = torch.randint(0, K, (B,), device=device)

                    xt = ref.sample_x_t(x0_batch, x1_batch, n, K)

                    loss = self.updater.train_forward_step(xt, x1_batch, n, K)
                    pbar.set_postfix(loss=f"{loss:.4f}")

            # --- Backward update stage ---
            print("Sampling new coupling (x0, x1) using forward model...")
            new_x0 = all_x0_real
            new_x1, _ = self.sampler.simulate(
                x_init=new_x0,
                model=self.updater.forward_model,
                direction="forward"
            )

            back_coupling_loader = DataLoader(TensorDataset(new_x0, new_x1), batch_size=self.batch_size, shuffle=True)
            for epoch in range(self.epochs):
                pbar = tqdm(back_coupling_loader, desc=f"L={l} | Backward Training")
                for x0_batch, x1_batch in pbar:
                    B = x0_batch.shape[0]
                    n = torch.randint(1, K + 1, (B,), device=device)

                    xt = ref.sample_x_t(x0_batch, x1_batch, n, K)

                    loss = self.updater.train_backward_step(xt, x0_batch, n, K)
                    pbar.set_postfix(loss=f"{loss:.4f}")