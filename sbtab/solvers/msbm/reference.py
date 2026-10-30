"""Exact discrete-time categorical reference for the shared-network MSBM.

``alpha`` retains its feature/tuning PER-STEP meaning. Ordered columns retain
that implementation's row-normalised Gaussian kernel (not the symmetric kernel
of canonical CSBM). Every transition is now a true power of the same matrix.
Log-space conditioning preserves bridges with extremely rare endpoint pairs.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from sbtab.bridge.reference import IncompatibleBridgeError
from sbtab.numerics import require_finite


@dataclass
class CategoricalReference:
    cardinalities: list[int]
    is_ordered: torch.Tensor
    total_number_of_q_powers: int
    alpha: float = 0.05
    device: torch.device = torch.device("cpu")
    dtype: torch.dtype = torch.float32  # retained API slot; bridge arithmetic uses float64

    def __post_init__(self) -> None:
        self.device = torch.device(self.device)
        self.cardinalities = [int(c) for c in self.cardinalities]
        if not self.cardinalities or any(c < 1 for c in self.cardinalities):
            raise ValueError("cardinalities must be nonempty and positive")
        self.D, self.S_max = len(self.cardinalities), max(self.cardinalities)
        self.S = torch.tensor(self.cardinalities, device=self.device)
        self.is_ordered = torch.as_tensor(self.is_ordered, dtype=torch.bool, device=self.device).reshape(-1)
        if len(self.is_ordered) != self.D:
            raise ValueError("is_ordered must have one entry per column")
        if self.total_number_of_q_powers < 1 or int(self.total_number_of_q_powers) != self.total_number_of_q_powers:
            raise ValueError("total_number_of_q_powers must be a positive integer")
        self.valid_mask = torch.arange(self.S_max, device=self.device)[None, None, :] < self.S[None, :, None]
        self.update_alpha(self.alpha)

    @staticmethod
    def _log_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # Chunk the first axis: a full S x S x S temporary is unnecessary.
        return torch.cat([
            torch.logsumexp(a[i:i + 16, :, None] + b[None, :, :], dim=1)
            for i in range(0, len(a), 16)
        ])

    def update_alpha(self, new_alpha: float) -> None:
        if not math.isfinite(new_alpha) or not 0 < new_alpha <= 1:
            raise ValueError("alpha must lie in (0, 1]")
        self.alpha = float(new_alpha)
        self._log_powers = []
        for d, S in enumerate(self.cardinalities):
            eye = torch.eye(S, device=self.device, dtype=torch.float64)
            powers = [eye.log()]
            if bool(self.is_ordered[d]) and S > 1:
                idx = torch.arange(S, device=self.device, dtype=torch.float64)
                logits = -4 * (idx[:, None] - idx[None, :]).square() / (self.alpha * (S - 1)) ** 2
                log_q = torch.log_softmax(logits, dim=-1)
                # Use ordinary matrix multiplication when all entries are safely
                # representable; otherwise keep even subnormal paths in log space.
                if float(log_q.min()) > -600:
                    q, current = log_q.exp(), eye
                    for _ in range(self.total_number_of_q_powers):
                        current = current @ q
                        powers.append(current.log())
                else:
                    powers.append(log_q)
                    for _ in range(1, self.total_number_of_q_powers):
                        powers.append(self._log_matmul(powers[-1], log_q))
            else:
                for k in range(1, self.total_number_of_q_powers + 1):
                    if S == 1:
                        qk = eye
                    else:
                        b = self.alpha * S / (S - 1)
                        if b < 1:
                            # Keep relative accuracy for very small jump rates.
                            log_retention = k * math.log1p(-b)
                            retention = math.exp(log_retention)
                            jump = -math.expm1(log_retention) / S
                        else:
                            retention = (1 - b) ** k
                            jump = (1 - retention) / S
                        qk = (retention * eye + jump).clamp_min(0)
                    powers.append(qk.log())
            self._log_powers.append(torch.stack(powers))

    @property
    def _powers(self) -> torch.Tensor:
        """Dense probability view for diagnostics; inference uses log powers."""
        return torch.stack([F.pad(q.exp(), (0, self.S_max - q.shape[-1], 0, self.S_max - q.shape[-1]))
                            for q in self._log_powers])

    def _index(self, n, B: int, lo: int, hi: int) -> torch.Tensor:
        n = torch.as_tensor(n, device=self.device)
        if n.is_floating_point() and not torch.equal(n, n.trunc()):
            raise ValueError("state indices must be integers")
        n = n.long().reshape(-1)
        if n.numel() == 1:
            n = n.expand(B)
        if n.numel() != B or bool(((n < lo) | (n > hi)).any()):
            raise ValueError(f"state indices must have one entry per row in [{lo}, {hi}]")
        return n

    def _steps(self, K: int) -> int:
        if K != self.total_number_of_q_powers:
            raise ValueError("total_steps must match the reference grid")
        return K

    @staticmethod
    def _normalise(log_w: torch.Tensor, **context) -> torch.Tensor:
        # Structural zeros are -inf in log space; NaN/+inf indicate numerical failure.
        require_finite(log_w.masked_fill(torch.isneginf(log_w), 0.0), "categorical_log_weights",
                       model="mixedsbm", **context)
        z = torch.logsumexp(log_w, dim=-1, keepdim=True)
        if torch.isneginf(z).any():
            raise IncompatibleBridgeError("The categorical reference cannot realise the requested endpoint pair")
        require_finite(z, "categorical_log_normalizer", model="mixedsbm", **context)
        return log_w - z

    def _stack(self, columns: list[torch.Tensor], *, log: bool = False) -> torch.Tensor:
        return torch.stack([F.pad(p if log else p.exp(), (0, self.S_max - p.shape[-1]),
                                  value=float("-inf") if log else 0.0) for p in columns], dim=1)

    def bridge_at_time(self, x_start, x_target, t, total_steps):
        K = self._steps(total_steps)
        n = self._index(t, len(x_start), 0, K)
        columns = []
        for d, q in enumerate(self._log_powers):
            log_w = q[n, x_start[:, d], :] + q[K - n, :, x_target[:, d]]
            columns.append(self._normalise(log_w))
        return self._stack(columns)

    def bridge_next_given_prev(self, x_t, x_target, n, K):
        K = self._steps(K)
        n = self._index(n, len(x_t), 0, K - 1)
        return self._stack([self._normalise(q[1, x_t[:, d], :] + q[K - n - 1, :, x_target[:, d]])
                            for d, q in enumerate(self._log_powers)])

    def bridge_prev_given_next(self, x_start, x_t, n):
        n = self._index(n, len(x_t), 1, self.total_number_of_q_powers)
        return self._stack([self._normalise(q[n - 1, x_start[:, d], :] + q[1].T[x_t[:, d]])
                            for d, q in enumerate(self._log_powers)])

    def _model_step(self, model_logits, x_t, n, *, forward: bool, log: bool):
        K = self.total_number_of_q_powers
        n = self._index(n, len(x_t), 0 if forward else 1, K - 1 if forward else K)
        columns = []
        for d, q in enumerate(self._log_powers):
            S = self.cardinalities[d]
            logits = model_logits[:, d, :S].double()
            context = dict(stage="categorical_transition", direction="f" if forward else "b", column=d)
            require_finite(logits.masked_fill(torch.isneginf(logits), 0.0), "categorical_logits",
                           model="mixedsbm", **context)
            log_p = torch.log_softmax(logits, dim=-1)
            require_finite(log_p.masked_fill(torch.isneginf(log_p), 0.0), "endpoint_log_probabilities",
                           model="mixedsbm", **context)
            reach = q[K - n, x_t[:, d], :] if forward else q[n, :, x_t[:, d]]
            # Periodic/reducible kernels can have genuinely unreachable endpoints.
            reachable = torch.isfinite(reach)
            log_ratio = log_p - torch.where(reachable, reach, torch.zeros_like(reach))
            log_ratio = log_ratio.masked_fill(~reachable, float("-inf"))
            if forward:
                terms = log_ratio[:, None, :] + q[K - n - 1]
                step = q[1, x_t[:, d], :]
            else:
                terms = log_ratio[:, None, :] + q[n - 1].transpose(-1, -2)
                step = q[1].T[x_t[:, d]]
            # An impossible intermediate state has all -inf summands. Avoid the
            # undefined logsumexp gradient there before assigning zero mass.
            possible = torch.isfinite(terms).any(dim=-1)
            safe_terms = torch.where(possible[..., None], terms, torch.zeros_like(terms))
            future = torch.logsumexp(safe_terms, dim=-1).masked_fill(~possible, float("-inf"))
            columns.append(self._normalise(step + future, **context))
        return self._stack(columns, log=log)

    def model_induced_next_step(self, model_logits, x_t, n, K):
        self._steps(K)
        return self._model_step(model_logits, x_t, n, forward=True, log=False)

    def model_induced_prev_step(self, model_logits, x_t, n):
        return self._model_step(model_logits, x_t, n, forward=False, log=False)

    def log_model_induced_next_step(self, model_logits, x_t, n, K):
        self._steps(K)
        return self._model_step(model_logits, x_t, n, forward=True, log=True)

    def log_model_induced_prev_step(self, model_logits, x_t, n):
        return self._model_step(model_logits, x_t, n, forward=False, log=True)

    def sample_from_probs(self, probs, generator: Optional[torch.Generator] = None):
        if not torch.isfinite(probs).all() or (probs < 0).any():
            raise ValueError("categorical probabilities must be finite and nonnegative")
        if (probs.masked_select(~self.valid_mask.expand_as(probs)) != 0).any():
            raise ValueError("probability mass on padded categories")
        if not torch.allclose(probs.sum(dim=-1), torch.ones_like(probs[..., 0]), atol=1e-7, rtol=1e-7):
            raise ValueError("categorical probabilities must sum to one")
        return torch.multinomial(probs.reshape(-1, self.S_max), 1, generator=generator).reshape(probs.shape[:2])

    def sample_x_t(self, x_start, x_target, t, total_steps, generator=None):
        return self.sample_from_probs(self.bridge_at_time(x_start, x_target, t, total_steps), generator)

    def sample_step(self, x_t, x_target, n, total_steps):
        return self.sample_from_probs(self.bridge_next_given_prev(x_t, x_target, n, total_steps))

    def describe(self) -> dict:
        return {"family": "feature_tuning_step_kernel", "transition_powers": "exact_log_domain",
                "ordered_kernel": "row_normalised_gaussian", "alpha": self.alpha,
                "num_steps": self.total_number_of_q_powers, "cardinalities": list(self.cardinalities),
                "is_ordered": self.is_ordered.tolist()}
