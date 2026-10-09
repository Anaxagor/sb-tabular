"""Unit-time, chunked sampler for the feature/tuning MSBM implementation."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Literal
import torch
from sbtab.bridge.timegrid import TimeGrid
from sbtab.bridge.sde import EulerMaruyama
from .reference import CategoricalReference

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
            generator: Optional[torch.Generator] = None,
            noise: Optional[bool] = None,
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
        if direction not in ("f", "b"):
            raise ValueError("direction must be 'f' or 'b'")
        integrator = self.integrator
        if integrator is not None and noise is not None:
            integrator = EulerMaruyama(noise=noise, sigma=integrator.sigma)
        was_training = model.training
        model.eval()

        device = x_cont_init.device if self.has_cont else x_cat_init.device
        K = self.timegrid.num_steps

        dt = torch.tensor(1.0 / K, device=device, dtype=torch.float32)

        total_samples = x_cont_init.shape[0] if self.has_cont else x_cat_init.shape[0]
        bs = batch_size if batch_size is not None else total_samples

        b_gen = generator
        if b_gen is None and seed is not None:
            b_gen = torch.Generator(device=device).manual_seed(int(seed))

        result_cont, result_cat = [], []
        path_cont_chunks, path_cat_chunks = ([], []) if return_path else (None, None)

        try:
            for i in range(0, total_samples, bs):
                b_cont = x_cont_init[i: i + bs].clone()
                b_cat = x_cat_init[i: i + bs].clone()
                b_size = b_cont.shape[0] if self.has_cont else b_cat.shape[0]

                if return_path:
                    b_path_cont = [b_cont.clone().cpu()]
                    b_path_cat = [b_cat.clone().cpu()]

                ks = range(K) if direction == "f" else range(K, 0, -1)

                for k in ks:
                    if direction == "f":
                        tau = float(k) / float(K)
                    else:
                        tau = 1.0 - float(K - k) / float(K)

                    tk = torch.full((b_size, 1), tau, device=device, dtype=torch.float32)

                    v_num, logits_cat = model(b_cont, b_cat, tk)

                    if self.has_cont:
                        b_cont = integrator.step(b_cont, drift=v_num, gamma=dt, generator=b_gen)

                    if self.has_cat:
                        if direction == "f":
                            probs = self.reference.model_induced_next_step(logits_cat, b_cat, k, K)
                        else:
                            probs = self.reference.model_induced_prev_step(logits_cat, b_cat, k)
                        b_cat = self.reference.sample_from_probs(probs, generator=b_gen)

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
        final_cat = torch.cat(result_cat, dim=0) if self.has_cat else torch.empty((total_samples, 0), dtype=torch.long,
                                                                                  device=device)

        paths = None
        if return_path:
            paths = {
                "cont": torch.cat(path_cont_chunks, dim=1) if self.has_cont else torch.empty((K + 1, total_samples, 0)),
                "cat": torch.cat(path_cat_chunks, dim=1) if self.has_cat else torch.empty((K + 1, total_samples, 0),
                                                                                          dtype=torch.long),
            }

        return final_cont, final_cat, paths
