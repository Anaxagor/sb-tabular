
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple, Dict

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
    """
    Simulates sample path trajectories for pure categorical variables on a defined time grid.

    Attributes:
        timegrid (TimeGrid): Object managing discretization time grid boundaries.
        reference (CategoricalReference): System discrete distribution transitions helper.
    """
    timegrid: TimeGrid
    reference: CategoricalReference

    @torch.no_grad()
    def simulate(
            self,
            x_init: torch.Tensor,
            model: torch.nn.Module,
            direction: Literal["forward", "backward"],
            return_path: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Simulates step-by-step categorical feature state updates along the time grid path.

        Args:
            x_init (torch.Tensor): Initial categorical indices distribution configuration (Shape: [B, D]).
            model (torch.nn.Module): Backbone network evaluating conditional logit lookaheads.
            direction (Literal["forward", "backward"]): Current solving flow propagation direction.
            return_path (bool, optional): If True, returns full trajectory history array. Defaults to False.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]:
                - Final states tensor matching terminating boundary rules (Shape: [B, D]).
                - Complete stack array containing history slices if return_path is enabled (Shape: [K+1, B, D]), else None.
        """
        was_training = model.training
        model.eval()

        device = self.reference.device
        x_init = torch.as_tensor(x_init, device=device, dtype=torch.long)

        K = self.timegrid.num_steps
        t_vals = self.timegrid.times().to(device)

        x = x_init.clone()
        curr_batch_size = x.shape[0]
        path = [x.clone()] if return_path else None

        if direction == "forward":
            ks = range(K)
        else:
            ks = range(K, 0, -1)

        try:
            for k in ks:
                t_idx = k if direction == "forward" else (k - 1)
                tk = t_vals[t_idx].expand(curr_batch_size, 1)
                logits = model(x, tk)

                if direction == "forward":
                    probs_step = self.reference.model_induced_next_step(
                        model_logits=logits, x_t=x, n=k, K=K
                    )
                else:
                    probs_step = self.reference.model_induced_prev_step(
                        model_logits=logits, x_t=x, n=k
                    )
                x = self.reference.sample_from_probs(probs_step)

                if return_path:
                    path.append(x.clone())
        finally:
            model.train(was_training)

        return x, (torch.stack(path) if return_path else None)

@dataclass
class MixedPathSampler:
    """
    Simulates coupled sample path updates for mixed continuous and categorical datasets.

    Handles continuous parts via an SDE integrator step (e.g., Euler-Maruyama) and
    categorical indices transitions via marginalized expectations.

    Attributes:
        timegrid (TimeGrid): Grid containing discretized tracking locations.
        reference (Optional[CategoricalReference]): Component tracking discrete features transitions.
        integrator (Optional[EulerMaruyama]): Numerical differential equation updater engine.
        has_cont (bool): Indicates presence of active continuous data dimensions. Defaults to True.
        has_cat (bool): Indicates presence of active categorical data dimensions. Defaults to True.
    """
    timegrid: TimeGrid
    reference: Optional[CategoricalReference]
    integrator: Optional[EulerMaruyama]
    has_cont: bool = True
    has_cat: bool = True

    @torch.no_grad()
    def simulate(
            self,
            x_cont_init: torch.Tensor,
            x_cat_init: torch.Tensor,
            model: torch.nn.Module,
            direction: Literal["f", "b"],
            return_path: bool = False,
            seed: Optional[int] = None,
            batch_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        Simulates state updates for mixed datatypes across the discretized timeline grid.

        Supports chunked micro-batch processing to prevent memory allocation overflows.

        Args:
            x_cont_init (torch.Tensor): Continuous features initialization coordinates (Shape: [B, cont_dim]).
            x_cat_init (torch.Tensor): Categorical features initialization indices (Shape: [B, cat_dim]).
            model (torch.nn.Module): Mixed feature multi-head estimation network backbone.
            direction (Literal["f", "b"]): Process direction assignment indicator ('f' for forward, 'b' for backward).
            return_path (bool, optional): If True, accumulates and returns state history dictionaries. Defaults to False.
            seed (Optional[int], optional): Random state configuration tracking key value. Defaults to None.
            batch_size (Optional[int], optional): Internal subset batch division evaluation constraint. Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
                - Generated continuous space terminal realizations tensor (Shape: [B, cont_dim]).
                - Generated categorical space terminal class assignments index tensor (Shape: [B, cat_dim]).
                - Dictionary containing structural logs ("cont", "cat") tracking historical steps if return_path is enabled, else None.
        """
        was_training = model.training
        model.eval()

        device = x_cont_init.device if self.has_cont else x_cat_init.device
        K = self.timegrid.num_steps
        t_vals = self.timegrid.times().to(device)
        g = self.timegrid.gammas().to(device)

        total_samples = x_cont_init.shape[0] if self.has_cont else x_cat_init.shape[0]
        bs = batch_size if batch_size is not None else total_samples

        result_cont, result_cat = [], []
        path_cont_chunks, path_cat_chunks = ([], []) if return_path else (None, None)

        try:
            for i in range(0, total_samples, bs):
                b_cont = x_cont_init[i: i + bs].clone()
                b_cat = x_cat_init[i: i + bs].clone()
                b_size = b_cont.shape[0] if self.has_cont else b_cat.shape[0]

                b_gen = None
                if seed is not None and self.has_cont:
                    b_gen = torch.Generator(device=device)
                    b_gen.manual_seed(int(seed) + i)

                if return_path:
                    b_path_cont = [b_cont.clone().cpu()]
                    b_path_cat = [b_cat.clone().cpu()]

                ks = range(K) if direction == "f" else range(K, 0, -1)

                for k in ks:
                    t_idx = k if direction == "f" else k - 1
                    tk = t_vals[t_idx].expand(b_size, 1)

                    v_num, logits_cat = model(b_cont, b_cat, tk)

                    if self.has_cont:
                        b_cont = self.integrator.step(b_cont, drift=v_num, gamma=g[t_idx], generator=b_gen)

                    if self.has_cat:
                        if direction == "f":
                            probs = self.reference.model_induced_next_step(logits_cat, b_cat, k, K)
                        else:
                            probs = self.reference.model_induced_prev_step(logits_cat, b_cat, k)
                        b_cat = self.reference.sample_from_probs(probs)

                    if return_path:
                        b_path_cont.append(b_cont.clone().cpu())
                        b_path_cat.append(b_cat.clone().cpu())

                result_cont.append(b_cont)
                result_cat.append(b_cat)

                if return_path:
                    path_cont_chunks.append(torch.stack(b_path_cont))
                    path_cat_chunks.append(torch.stack(b_path_cat))

        finally:
            model.train(was_training)

        final_cont = torch.cat(result_cont, dim=0) if self.has_cont else torch.empty((total_samples, 0), device=device)
        final_cat = torch.cat(result_cat, dim=0) if self.has_cat else torch.empty((total_samples, 0), dtype=torch.long, device=device)

        paths = None
        if return_path:
            paths = {
                "cont": torch.cat(path_cont_chunks, dim=1) if self.has_cont else torch.empty((K+1, total_samples, 0)),
                "cat": torch.cat(path_cat_chunks, dim=1) if self.has_cat else torch.empty((K+1, total_samples, 0), dtype=torch.long),
            }

        return final_cont, final_cat, paths