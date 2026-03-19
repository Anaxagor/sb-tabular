
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import torch

from .reference import CategoricalReference
from .timegrid import TimeGrid
from .sde import EulerMaruyama, FieldFn


@dataclass
class PathSampler:
    """
    Simulate trajectories on a TimeGrid given a field/drift function.

    direction:
      - "forward": k = 0..K-1 (increasing time)
      - "backward": k = K-1..0 (decreasing time)

    Returns:
      - x0 and full path optionally.
    """
    timegrid: TimeGrid
    integrator: EulerMaruyama

    def simulate(
        self,
        x_init: torch.Tensor,
        field: FieldFn,
        direction: Literal["forward", "backward"],
        return_path: bool = False,
        seed: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        g = self.timegrid.gammas()
        t = self.timegrid.times()
        K = self.timegrid.num_steps

        gen = None
        if seed is not None:
            gen = torch.Generator(device=str(x_init.device))
            gen.manual_seed(int(seed))

        x = x_init
        if return_path:
            path = torch.empty((K + 1, x.shape[0], x.shape[1]), device=x.device, dtype=x.dtype)
            path[0] = x

        if direction == "forward":
            ks = range(0, K)
        elif direction == "backward":
            ks = range(K - 1, -1, -1)
        else:
            raise ValueError(f"Unknown direction: {direction}")

        step_i = 0
        for k in ks:
            # Use time value t[k] and integer step index k
            tk = t[k].expand(x.shape[0], 1)
            kk = torch.full((x.shape[0],), int(k), device=x.device, dtype=torch.long)

            drift = field(x, tk, kk)
            x = self.integrator.step(x, drift=drift, gamma=g[k], generator=gen)

            if return_path:
                path[step_i + 1] = x
            step_i += 1

        return x, (path if return_path else None)

@dataclass
class DiscretePathSampler:
    timegrid: TimeGrid
    reference: CategoricalReference

    @torch.no_grad()
    def simulate(self, x_init: torch.Tensor, model: torch.nn.Module, direction: Literal["forward", "backward"],
                 return_path: bool = False):
        model.eval()
        K = self.timegrid.num_steps
        t_vals = self.timegrid.times()

        x = x_init.clone()
        curr_batch_size = x.shape[0]
        path = [x] if return_path else None

        ks = range(K) if direction == "forward" else range(K - 1, -1, -1)

        for k in ks:
            tk = t_vals[k].expand(curr_batch_size, 1).to(x.device)

            logits = model(x, tk)
            probs = torch.softmax(logits, dim=-1)

            flat_probs = probs.reshape(-1, self.reference.S_max)
            x_target_pred = torch.multinomial(flat_probs + 1e-12, 1).view(x.shape)

            if direction == "forward":
                x = self.reference.sample(
                    x_prev=x,
                    x_target=x_target_pred,
                    t=k,
                    total_steps=K
                )
            else:
                t_prev = k
                probs_back = self.reference.bridge_at_time(
                    x_start=x_target_pred,
                    x_target=x,
                    t=torch.full((curr_batch_size,), t_prev, device=x.device),
                    total_steps=k + 1
                )
                x = torch.multinomial(probs_back.view(-1, self.reference.S_max) + 1e-12, 1).view(x.shape)

            if return_path:
                path.append(x)

        model.train()
        return x, (torch.stack(path) if return_path else None)