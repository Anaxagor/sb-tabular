
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import torch

from .reference import CategoricalReference
from .timegrid import TimeGrid
from .sde import EulerMaruyama, FieldFn


def _make_generator(seed: Optional[int], device: torch.device) -> Optional[torch.Generator]:
    if seed is None:
        return None
    gen = torch.Generator(device=str(device))
    gen.manual_seed(int(seed))
    return gen


@dataclass
class PathSampler:
    """
    Simulate trajectories on a TimeGrid given a field/drift function.

    direction:
      - "forward": k = 0..K-1 (increasing time)
      - "backward": k = K-1..0 (decreasing time)

    Legacy DSB clock: the field receives ``timegrid.times()[k]`` (the END of
    interval k) together with the integer interval index k.

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

        gen = _make_generator(seed, x_init.device)

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


# ---------------------------------------------------------------------------
# Bridge-matching samplers (CSBM, MixedSBM).
#
# One clock for everything. The state with index n lives at grid()[n]:
#
#   forward  step n -> n+1, n = 0..N-1 : the network sees t[n], the continuous
#            update uses dt[n], the categorical update is the step from index n.
#   backward step n -> n-1, n = N..1   : the network sees t[n], the continuous
#            update uses dt[n-1], the categorical update is the step from index n.
#
# In both directions the network input is the time of the CURRENT state, which
# is exactly what training feeds it. Forward starts at n = 0 and backward at
# n = N, so the two directions need different endpoint indices.
# ---------------------------------------------------------------------------
@dataclass
class DiscretePathSampler:
    timegrid: TimeGrid
    reference: CategoricalReference

    @torch.no_grad()
    def simulate(self, x_init: torch.Tensor, model: torch.nn.Module, direction: Literal["forward", "backward"],
                 return_path: bool = False, seed: Optional[int] = None,
                 generator: Optional[torch.Generator] = None):
        if direction not in ("forward", "backward"):
            raise ValueError(f"Unknown direction: {direction}")
        was_training = model.training
        model.eval()
        N = self.timegrid.num_steps
        grid = self.timegrid.grid().to(x_init.device)
        gen = generator if generator is not None else _make_generator(seed, x_init.device)

        x = x_init.clone()
        B = x.shape[0]
        path = [x.clone()] if return_path else None

        ns = range(N) if direction == "forward" else range(N, 0, -1)
        for n in ns:
            tn = grid[n].expand(B, 1)
            logits = model(x, tn)
            if direction == "forward":
                probs_step = self.reference.model_induced_next_step(model_logits=logits, x_t=x, n=n)
            else:
                probs_step = self.reference.model_induced_prev_step(model_logits=logits, x_t=x, n=n)
            x = self.reference.sample_from_probs(probs_step, generator=gen)

            if return_path:
                path.append(x.clone())

        model.train(was_training)
        return x, (torch.stack(path) if return_path else None)


@dataclass
class MixedPathSampler:
    timegrid: TimeGrid
    reference: Optional[CategoricalReference]
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
            generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[dict]]:
        if direction not in ("forward", "backward"):
            raise ValueError(f"Unknown direction: {direction}")
        was_training = model.training
        model.eval()
        N = self.timegrid.num_steps
        grid = self.timegrid.grid().to(x_cont_init.device)
        dt = self.timegrid.dt().to(x_cont_init.device)
        gen = generator if generator is not None else _make_generator(seed, x_cont_init.device)

        x_cont = x_cont_init.clone()
        x_cat = x_cat_init.clone()
        B = x_cont.shape[0]
        has_cont = x_cont.shape[1] > 0
        has_cat = x_cat.shape[1] > 0 and self.reference is not None

        path_cont, path_cat = None, None
        if return_path:
            path_cont = [x_cont.clone()]
            path_cat = [x_cat.clone()]

        ns = range(N) if direction == "forward" else range(N, 0, -1)
        for n in ns:
            step = dt[n] if direction == "forward" else dt[n - 1]
            tn = grid[n].expand(B, 1)

            v_num, logits_cat = model(x_cont, x_cat, tn)

            if has_cont:
                x_cont = self.integrator.step(x_cont, drift=v_num, gamma=step, generator=gen)

            if has_cat:
                if direction == "forward":
                    probs_step = self.reference.model_induced_next_step(model_logits=logits_cat, x_t=x_cat, n=n)
                else:
                    probs_step = self.reference.model_induced_prev_step(model_logits=logits_cat, x_t=x_cat, n=n)
                x_cat = self.reference.sample_from_probs(probs_step, generator=gen)

            if return_path:
                path_cont.append(x_cont.clone())
                path_cat.append(x_cat.clone())

        model.train(was_training)

        paths = None
        if return_path:
            paths = {
                "cont": torch.stack(path_cont),
                "cat": torch.stack(path_cat)
            }

        return x_cont, x_cat, paths
