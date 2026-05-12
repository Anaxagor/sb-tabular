
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
        path = [x.clone()] if return_path else None

        if direction == "forward":
            ks = range(K)
        else:
            ks = range(K, 0, -1)

        for k in ks:
            t_idx = k if direction == "forward" else (k - 1)
            tk = t_vals[t_idx].expand(curr_batch_size, 1).to(x.device)
            logits = model(x, tk)

            if direction == "forward":
                probs_step = self.reference.model_induced_next_step(
                    model_logits=logits,
                    x_t=x,
                    n=k,
                    K=K
                )
                x = self.reference.sample_from_probs(probs_step)
            else:
                probs_step = self.reference.model_induced_prev_step(
                    model_logits=logits,
                    x_t=x,
                    n=k
                )
                x = self.reference.sample_from_probs(probs_step)

            if return_path:
                path.append(x.clone())

        model.train()
        return x, (torch.stack(path) if return_path else None)

@dataclass
class MixedPathSampler:
    timegrid: TimeGrid
    reference: CategoricalReference
    integrator: EulerMaruyama

    @torch.no_grad()
    def simulate(
            self,
            x_cont_init: torch.Tensor,
            x_cat_init: torch.Tensor,
            model: torch.nn.Module,
            direction: Literal["forward", "backward"],
            return_path: bool = False,
            seed: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[dict]]:
        model.eval()
        K = self.timegrid.num_steps
        t_vals = self.timegrid.times()

        gen = None
        if seed is not None:
            gen = torch.Generator(device=str(x_cont_init.device))
            gen.manual_seed(int(seed))

        x_cont = x_cont_init.clone()
        x_cat = x_cat_init.clone()
        B = x_cont.shape[0]
        g = self.timegrid.gammas()

        path_cont, path_cat = None, None
        if return_path:
            path_cont = [x_cont.clone()]
            path_cat = [x_cat.clone()]

        if direction == "forward":
            ks = range(K)
        else:
            ks = range(K, 0, -1)

        for k in ks:
            if direction == "forward":
                t_idx = k
            else:
                t_idx = k - 1

            tk = t_vals[t_idx].expand(B, 1).to(x_cont.device)

            v_num, logits_cat = model(x_cont, x_cat, tk)

            drift = v_num
            x_cont = self.integrator.step(x_cont, drift=drift, gamma=g[t_idx], generator=gen)

            if direction == "forward":
                probs_step = self.reference.model_induced_next_step(
                    model_logits=logits_cat,
                    x_t=x_cat,
                    n=k,
                    K=K
                )
            else:
                probs_step = self.reference.model_induced_prev_step(
                    model_logits=logits_cat,
                    x_t=x_cat,
                    n=k
                )

            probs_step = torch.nan_to_num(probs_step, nan=0.0)

            row_sums = probs_step.sum(dim=-1, keepdim=True)
            if (row_sums <= 0).any():
                print("Some sums of probs equal to 0!")
                probs_step = probs_step + 1e-10

            x_cat = self.reference.sample_from_probs(probs_step)

            if return_path:
                path_cont.append(x_cont.clone())
                path_cat.append(x_cat.clone())

        model.train()

        paths = None
        if return_path:
            paths = {
                "cont": torch.stack(path_cont),
                "cat": torch.stack(path_cat)
            }

        return x_cont, x_cat, paths