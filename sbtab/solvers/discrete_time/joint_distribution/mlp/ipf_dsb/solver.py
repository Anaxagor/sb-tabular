"""
Discrete-time joint IPF-DSB with PER-STEP MLPs (registry id ``dsb_dt_joint_mlp``).

The IPF algorithm (declared OU/Brownian reference, full-trajectory stochastic
caches, mean-matching displacement targets, displacement-adding sampler,
checkpointing) is shared with, and documented in,
``sbtab.solvers.continuous_time.joint_distribution.mlp.ipf_dsb.solver``.

What makes this solver discrete-time is the parameterisation: there is ONE
independent plain MLP per edge k = 0..K-1 and per direction (2 K networks, no
time input, no parameter sharing between edges). Edge k joins X_k and X_{k+1};
``net_f.steps[k]`` is the forward displacement F_k(x) - x evaluated at X_k and
``net_b.steps[k]`` the backward displacement B_k(x) - x evaluated at X_{k+1}.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List

import torch
from torch import nn

from sbtab.models.neural.mlp import StepIndexedMLP, StepMLPConfig
from sbtab.numerics import check_gradients
from sbtab.solvers.continuous_time.joint_distribution.mlp.ipf_dsb.solver import (
    CHECKPOINT_FORMAT,
    IPFCache,
    IPFDSBConfig as _TimeConditionedIPFDSBConfig,
    IPFDSBSolver as _TimeConditionedIPFDSBSolver,
)

__all__ = ["CHECKPOINT_FORMAT", "IPFCache", "IPFDSBConfig", "IPFDSBSolver"]


@dataclass
class IPFDSBConfig(_TimeConditionedIPFDSBConfig):
    """
    Same fields as the time-conditioned config (reference, grid, noise, seed ...),
    with these differences:

      - ``hidden_units`` / ``n_layers`` / ``dropout`` size EACH of the 2 K per-edge
        MLPs (smaller defaults: 128 x 3).
      - ``batch_size`` is the number of TRAJECTORIES per update: every update
        trains all K edge networks, each on ``batch_size`` rows of its own edge.
        A cache holds ``cache_batches * batch_size`` trajectories, so an epoch is
        ``cache_batches`` updates (K times the rows of the time-conditioned solver).
      - ``grad_clip`` is applied per edge network.
      - ``time_features`` and ``time_scale`` are IGNORED: per-step networks take
        no time input. They are kept for config compatibility.
    """
    hidden_units: int = 128
    n_layers: int = 3


class IPFDSBSolver(_TimeConditionedIPFDSBSolver):
    """
    IPF-DSB with genuinely PER-STEP networks (``time_parameterization`` =
    "per_step"): net_f / net_b are ``StepIndexedMLP`` with one MLP per edge.

    Invariants (asserted):
      - every half-iteration updates every edge network k = 0..K-1 of the trained
        direction (``stage_log[i]["edge_updates"]``);
      - every simulation / sampling sweep evaluates every edge network exactly in
        grid order with the index of the edge whose gamma_k is used.

    Public API: identical to the time-conditioned solver (fit, sample,
    sample_paths, save_checkpoint, load_checkpoint, describe, variant_id,
    n_updates, stage_log).
    """

    canonical_id = "dsb_dt_joint_mlp"
    time_parameterization = "per_step"
    config_class = IPFDSBConfig

    # ------------------------------------------------------------------ networks
    def _build_network(self) -> nn.Module:
        cfg = self.cfg
        return StepIndexedMLP(StepMLPConfig(
            in_dim=self.dim,
            num_steps=self.K,
            hidden_dim=cfg.hidden_units,
            n_layers=cfg.n_layers,
            dropout=cfg.dropout,
        ))

    def _displacement(self, net: nn.Module, direction: str, x: torch.Tensor, k) -> torch.Tensor:
        """d(x, k) from the k-th edge network; all rows of x belong to edge k."""
        if isinstance(k, torch.Tensor):
            raise TypeError("per-step networks are evaluated one edge at a time (int k)")
        k = int(k)
        assert 0 <= k < self.K, f"edge index {k} outside 0..{self.K - 1}"
        return net(x, k)

    # ------------------------------------------------------------------ training
    def _n_trajectories(self) -> int:
        return int(self.cfg.cache_batches * self.cfg.batch_size)

    def _iter_batches(self, cache: IPFCache, gen: torch.Generator) -> Iterator[tuple]:
        """Shuffled minibatches of trajectories: (K, m, D) inputs and targets; partial batches kept."""
        M = cache.x.shape[1]
        perm = torch.randperm(M, generator=gen, device=self.device)
        bs = int(self.cfg.batch_size)
        for s in range(0, M, bs):
            idx = perm[s:s + bs]
            yield cache.x[:, idx], cache.y[:, idx]

    def _batch_loss(self, net: nn.Module, direction: str, batch: tuple) -> torch.Tensor:
        # Sum (not mean) over edges: the gradient of edge k is that of its own MSE,
        # so the K regressions are independent.
        xb, yb = batch
        assert xb.shape[0] == self.K
        total = xb.new_zeros(())
        for k in range(self.K):
            total = total + self.loss(self._displacement(net, direction, xb[k], k), yb[k])
        return total

    def _clip_gradients(self, net: nn.Module, **context) -> None:
        # Reject every bad edge before modifying any gradients; keep per-edge clipping.
        check_gradients(net.parameters(), **context)
        if self.cfg.grad_clip is not None:
            for edge, step_net in enumerate(net.steps):
                check_gradients(step_net.parameters(), max_norm=self.cfg.grad_clip, edge=edge, **context)

    def _train_phase(self, iteration, direction, simulated_with, x_data, gen) -> dict:
        self._edge_updates: List[int] = [0] * self.K
        return super()._train_phase(iteration, direction, simulated_with, x_data, gen)

    def _record_update(self, direction: str, batch: tuple) -> None:
        xb, _ = batch
        for k in range(self.K):
            if xb[k].shape[0] > 0:
                self._edge_updates[k] += 1

    def _check_phase(self, direction: str, entry: dict) -> None:
        untrained = [k for k, n in enumerate(self._edge_updates) if n <= 0]
        if untrained:
            raise AssertionError(f"{direction} edge networks {untrained} received no update")
        entry["edge_updates"] = list(self._edge_updates)

    # ------------------------------------------------------------------ metadata
    def _describe_clock(self):
        return None  # no time input: the edge index selects the network
