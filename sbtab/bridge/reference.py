
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import torch

from .timegrid import TimeGrid
from sbtab.numerics import require_finite


@dataclass
class GaussianReference:
    """
    Simple Gaussian reference distribution/process endpoints.

    In DSB/IPF practice for tabular:
      - terminal distribution at t=T is often standard Gaussian
      - initial distribution at t=0 is data distribution

    This class provides sampling for the "noise/prior" endpoint.
    """
    dim: int
    mean: float = 0.0
    std: float = 1.0
    device: Optional[torch.device] = None
    dtype: torch.dtype = torch.float32

    def sample(
        self,
        n: int,
        seed: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        if n <= 0:
            raise ValueError("n must be positive")
        dev = self.device or torch.device("cpu")
        g = generator
        if g is None and seed is not None:
            g = torch.Generator(device=str(dev))
            g.manual_seed(int(seed))
        # With neither a seed nor a generator the global RNG is used. A fresh
        # unseeded torch.Generator has a FIXED default seed, so building one here
        # would return identical draws on every call.
        x = torch.randn((n, self.dim), generator=g, device=dev, dtype=self.dtype)
        return x * self.std + self.mean


class IncompatibleBridgeError(ValueError):
    """
    The reference assigns zero probability to a requested endpoint pair, so the
    conditional bridge does not exist. This is reported instead of being hidden
    behind an epsilon and an effectively uniform fallback.
    """


# ---------------------------------------------------------------------------
# Per-column transition families. Every family is a time-homogeneous Markov
# semigroup Q(s) = exp(s R), so Q(a) Q(b) = Q(a + b) holds on any grid and the
# reference law does not depend on how many integration steps are used.
# ---------------------------------------------------------------------------
class _UniformKernel:
    """
    R = rate * (11^T / S - I)  =>  Q(s) = a I + (1 - a) 11^T / S,  a = exp(-rate s).

    Closed form; nothing of size S x S is stored, so high-cardinality nominal
    columns stay cheap.
    """

    def __init__(self, S: int, rate: float, grid: torch.Tensor, device: torch.device):
        self.S = int(S)
        self.rate = float(rate)
        dt = grid[1:] - grid[:-1]
        T = grid[-1]
        # retention a(s) and the off-diagonal mass (1 - a(s)) / S, the latter via
        # expm1 so that it keeps relative accuracy for small rate * s.
        self._a = {
            "from_start": torch.exp(-self.rate * grid),
            "to_end": torch.exp(-self.rate * (T - grid)),
            "step": torch.exp(-self.rate * dt),
        }
        self._u = {
            "from_start": -torch.expm1(-self.rate * grid) / self.S,
            "to_end": -torch.expm1(-self.rate * (T - grid)) / self.S,
            "step": -torch.expm1(-self.rate * dt) / self.S,
        }
        self._a = {k: v.to(device) for k, v in self._a.items()}
        self._u = {k: v.to(device) for k, v in self._u.items()}

    def row(self, kind: str, idx: torch.Tensor, i: torch.Tensor) -> torch.Tensor:
        a = self._a[kind][idx].unsqueeze(-1)
        u = self._u[kind][idx].unsqueeze(-1)
        row = u.expand(-1, self.S).clone()
        return row.scatter_add(1, i.unsqueeze(-1), a)

    col = row  # the kernel is symmetric

    def matvec(self, kind: str, idx: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        a = self._a[kind][idx].unsqueeze(-1)
        u = self._u[kind][idx].unsqueeze(-1)
        return a * v + u * v.sum(dim=-1, keepdim=True)

    vecmat = matvec

    def dense(self, kind: str, idx: int) -> torch.Tensor:
        a = self._a[kind][idx]
        u = self._u[kind][idx]
        return a * torch.eye(self.S, dtype=torch.float64, device=a.device) + u


class _OrderedKernel:
    """
    R = rate * (P - I) with P a symmetric, doubly stochastic, Gaussian-shaped
    jump matrix over the ORDERED support (D3PM discretised Gaussian):

        P[i, j] = exp(-4 (i - j)^2 / (bandwidth (S - 1))^2) / Z   (i != j)
        P[i, i] = 1 - sum_{j != i} P[i, j]
        Z       = sum_{n = -(S-1)}^{S-1} exp(-4 n^2 / (bandwidth (S - 1))^2)

    Q(s) = exp(s R) is evaluated by uniformisation,

        Q(s) = sum_m Poisson(m; rate s) P^m,

    a sum of non-negative terms. There is no cancellation, so tiny entries keep
    their *relative* accuracy; bridges between rare endpoint pairs are ratios of
    exactly such entries.
    """

    def __init__(self, S: int, rate: float, bandwidth: float, grid: torch.Tensor, device: torch.device):
        self.S = int(S)
        self.rate = float(rate)
        self.bandwidth = float(bandwidth)
        self.P = self.jump_matrix(self.S, self.bandwidth)

        dt = grid[1:] - grid[:-1]
        T = grid[-1]
        lam_max = self.rate * float(T)
        M = int(math.ceil(lam_max + 15.0 * math.sqrt(max(lam_max, 1.0)) + 50.0))
        powers = torch.empty((M + 1, self.S, self.S), dtype=torch.float64)
        powers[0] = torch.eye(self.S, dtype=torch.float64)
        for m in range(1, M + 1):
            powers[m] = powers[m - 1] @ self.P
        self._log_fact = torch.lgamma(torch.arange(M + 1, dtype=torch.float64) + 1.0)
        self._m = torch.arange(M + 1, dtype=torch.float64)

        self._stacks = {
            "from_start": self._expm(powers, grid).to(device),
            "to_end": self._expm(powers, T - grid).to(device),
            "step": self._expm(powers, dt).to(device),
        }

    @staticmethod
    def jump_matrix(S: int, bandwidth: float) -> torch.Tensor:
        if S == 1:
            return torch.ones((1, 1), dtype=torch.float64)
        idx = torch.arange(S, dtype=torch.float64)
        d2 = (idx.view(-1, 1) - idx.view(1, -1)) ** 2
        scale = (bandwidth * (S - 1)) ** 2
        n = torch.arange(-(S - 1), S, dtype=torch.float64)
        Z = torch.exp(-4.0 * n ** 2 / scale).sum()
        P = torch.exp(-4.0 * d2 / scale) / Z
        P.fill_diagonal_(0.0)
        P = P + torch.diag(1.0 - P.sum(dim=1))
        return P

    def _expm(self, powers: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        lam = (self.rate * s).clamp_min(0.0).view(-1, 1)  # (L, 1)
        logw = -lam + self._m.view(1, -1) * torch.log(lam.clamp_min(1e-300)) - self._log_fact.view(1, -1)
        w = torch.exp(logw)
        # lam == 0 is the identity: all Poisson mass sits on m = 0.
        zero = lam.view(-1) == 0
        if zero.any():
            w[zero] = 0.0
            w[zero, 0] = 1.0
        tail = 1.0 - w.sum(dim=1)
        if float(tail.abs().max()) > 1e-13:
            raise RuntimeError("uniformisation series truncated too early")
        w = w / w.sum(dim=1, keepdim=True)
        return torch.einsum("lm,mij->lij", w, powers)

    def row(self, kind: str, idx: torch.Tensor, i: torch.Tensor) -> torch.Tensor:
        return self._stacks[kind][idx, i, :]

    def col(self, kind: str, idx: torch.Tensor, j: torch.Tensor) -> torch.Tensor:
        return self._stacks[kind][idx, :, j]

    def matvec(self, kind: str, idx: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bsj,bj->bs", self._stacks[kind][idx], v)

    def vecmat(self, kind: str, idx: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bi,bis->bs", v, self._stacks[kind][idx])

    def dense(self, kind: str, idx: int) -> torch.Tensor:
        return self._stacks[kind][idx]


@dataclass
class CategoricalReference:
    """
    Reference Markov process for a block of categorical columns on a TimeGrid.

    States are indexed 0..N along ``timegrid.grid()``. For column d the reference
    is the semigroup Q_d(s) = exp(s R_d); the transition between state indices
    a <= b is Q_d(t[b] - t[a]). Three cached views are used:

        from_start[n] = Q(t[n])          n = 0..N     (index 0  -> n)
        to_end[n]     = Q(T - t[n])      n = 0..N     (index n  -> N)
        step[i]       = Q(dt[i])         i = 0..N-1   (index i  -> i+1)

    Because elapsed time alone determines a transition, changing the number of
    steps or using a non-uniform grid leaves the reference law unchanged.

    Parameters
      mixing_rate        jump rate per unit time (both kernels).
      ordered_bandwidth  width of the ordered jump kernel relative to S - 1.
      is_ordered         per-column flag. Set it ONLY for columns whose state
                         index follows a meaningful order (ordinal / numeric
                         discrete support). Nominal codes must use the uniform
                         kernel: their integer codes carry no order.

    All probabilities are float64. A bridge that the reference cannot realise
    raises IncompatibleBridgeError; no epsilon is added anywhere.
    """
    cardinalities: Sequence[int]
    is_ordered: torch.Tensor
    timegrid: TimeGrid
    mixing_rate: float = 1.0
    ordered_bandwidth: float = 0.2
    device: torch.device = torch.device("cpu")
    dtype: torch.dtype = torch.float64
    _kernels: List[object] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self):
        self.cardinalities = [int(c) for c in self.cardinalities]
        if any(c < 1 for c in self.cardinalities):
            raise ValueError("every cardinality must be >= 1")
        self.device = torch.device(self.device)
        self.D = len(self.cardinalities)
        self.is_ordered = torch.as_tensor(self.is_ordered).clone().detach().to(dtype=torch.bool, device="cpu").view(-1)
        if self.is_ordered.numel() != self.D:
            raise ValueError("is_ordered must have one entry per categorical column")
        self.S = torch.tensor(self.cardinalities, device=self.device, dtype=torch.long)
        self.S_max = int(max(self.cardinalities)) if self.D > 0 else 0
        self.N = int(self.timegrid.num_steps)
        self._build()

    # ------------------------------------------------------------------ kernels
    def _build(self) -> None:
        if not math.isfinite(self.mixing_rate) or not self.mixing_rate > 0:
            raise ValueError("mixing_rate must be positive")
        if not math.isfinite(self.ordered_bandwidth) or not self.ordered_bandwidth > 0:
            raise ValueError("ordered_bandwidth must be positive")
        grid = self.timegrid._grid64()
        self._kernels = []
        for d, S_d in enumerate(self.cardinalities):
            if bool(self.is_ordered[d]):
                k = _OrderedKernel(S_d, self.mixing_rate, self.ordered_bandwidth, grid, self.device)
            else:
                k = _UniformKernel(S_d, self.mixing_rate, grid, self.device)
            self._kernels.append(k)
        if self.D > 0:
            ar = torch.arange(self.S_max, device=self.device).view(1, -1)
            self.valid_mask = ar < self.S.view(-1, 1)  # (D, S_max)
        else:
            self.valid_mask = torch.zeros((0, 0), dtype=torch.bool, device=self.device)

    def set_kernel(self, mixing_rate: Optional[float] = None, ordered_bandwidth: Optional[float] = None) -> None:
        """Change the reference and rebuild EVERY cached transition."""
        if mixing_rate is not None:
            self.mixing_rate = float(mixing_rate)
        if ordered_bandwidth is not None:
            self.ordered_bandwidth = float(ordered_bandwidth)
        self._build()

    @staticmethod
    def rate_from_step_alpha(alpha: float, cardinality: int, num_steps: int, horizon: float = 1.0) -> float:
        """
        Mixing rate reproducing the legacy per-step uniform kernel
        Q = (1 - b) I + b 11^T / S with b = alpha S / (S - 1), applied
        ``num_steps`` times over ``horizon``:  exp(-rate * horizon) = (1 - b)^num_steps.
        """
        S = int(cardinality)
        if S < 2:
            raise ValueError("cardinality must be >= 2")
        b = alpha * S / (S - 1)
        if not 0 < b < 1:
            raise ValueError("alpha S / (S - 1) must lie in (0, 1)")
        return -num_steps * math.log1p(-b) / horizon

    def transition(self, d: int, kind: str, idx: int) -> torch.Tensor:
        """Dense (S_d, S_d) matrix; kind is 'from_start' | 'to_end' | 'step'."""
        return self._kernels[d].dense(kind, int(idx))

    def describe(self) -> dict:
        return {
            "family": "generator_exponential",
            "mixing_rate": float(self.mixing_rate),
            "ordered_bandwidth": float(self.ordered_bandwidth),
            "cardinalities": list(self.cardinalities),
            "is_ordered": [bool(b) for b in self.is_ordered.tolist()],
            "grid": [float(v) for v in self.timegrid._grid64().tolist()],
        }

    # ------------------------------------------------------------------ helpers
    def _index(self, n, batch_size: int, lo: int, hi: int, what: str) -> torch.Tensor:
        if not isinstance(n, torch.Tensor):
            n = torch.full((batch_size,), int(n), device=self.device, dtype=torch.long)
        n = n.to(device=self.device, dtype=torch.long)
        if n.dim() == 0:
            n = n.expand(batch_size)
        n = n.reshape(-1)
        if n.numel() != batch_size:
            raise ValueError("time index must be a scalar or have one entry per row")
        if n.numel() and (int(n.min()) < lo or int(n.max()) > hi):
            raise IndexError(f"{what}: state index must lie in [{lo}, {hi}]")
        return n

    def _normalise(self, w: torch.Tensor, d: int, what: str) -> torch.Tensor:
        context = dict(stage=what, column=d)
        require_finite(w, "categorical_weights", **context)
        Z = w.sum(dim=-1, keepdim=True)
        require_finite(Z, "categorical_normalizer", **context)
        bad = ~(Z.squeeze(-1) > 0)
        if bad.any():
            raise IncompatibleBridgeError(
                f"{what}: the reference gives zero probability to {int(bad.sum())} endpoint "
                f"configuration(s) in categorical column {d} (S={self.cardinalities[d]}, "
                f"ordered={bool(self.is_ordered[d])}, mixing_rate={self.mixing_rate}, "
                f"ordered_bandwidth={self.ordered_bandwidth}). Increase the mixing rate/bandwidth "
                f"or use a compatible coupling."
            )
        return w / Z

    def _pad(self, probs_d: torch.Tensor) -> torch.Tensor:
        S_d = probs_d.shape[-1]
        if S_d == self.S_max:
            return probs_d
        pad = torch.zeros((probs_d.shape[0], self.S_max - S_d), device=probs_d.device, dtype=probs_d.dtype)
        return torch.cat([probs_d, pad], dim=-1)

    def masked_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Set padded category logits to -inf so they carry exactly zero mass."""
        masked = logits.masked_fill(~self.valid_mask.unsqueeze(0), float("-inf"))
        require_finite(masked.masked_fill(torch.isneginf(masked), 0.0), "categorical_logits",
                       stage="categorical_transition")
        return masked

    # ------------------------------------------------------------------ bridges
    def bridge_at_time(self, x_start: torch.Tensor, x_target: torch.Tensor, n) -> torch.Tensor:
        """P(x_n = . | x_0, x_N) for n in [0, N]; shape (B, D, S_max)."""
        B = x_start.shape[0]
        n = self._index(n, B, 0, self.N, "bridge_at_time")
        out = []
        for d, k in enumerate(self._kernels):
            w = k.row("from_start", n, x_start[:, d]) * k.col("to_end", n, x_target[:, d])
            out.append(self._pad(self._normalise(w, d, "bridge_at_time")))
        return torch.stack(out, dim=1)

    def bridge_next_given_prev(self, x_t: torch.Tensor, x_target: torch.Tensor, n) -> torch.Tensor:
        """P(x_{n+1} = . | x_n, x_N) for n in [0, N-1]."""
        B = x_t.shape[0]
        n = self._index(n, B, 0, self.N - 1, "bridge_next_given_prev")
        out = []
        for d, k in enumerate(self._kernels):
            w = k.row("step", n, x_t[:, d]) * k.col("to_end", n + 1, x_target[:, d])
            out.append(self._pad(self._normalise(w, d, "bridge_next_given_prev")))
        return torch.stack(out, dim=1)

    def bridge_prev_given_next(self, x_start: torch.Tensor, x_t: torch.Tensor, n) -> torch.Tensor:
        """P(x_{n-1} = . | x_0, x_n) for n in [1, N]."""
        B = x_t.shape[0]
        n = self._index(n, B, 1, self.N, "bridge_prev_given_next")
        out = []
        for d, k in enumerate(self._kernels):
            w = k.row("from_start", n - 1, x_start[:, d]) * k.col("step", n - 1, x_t[:, d])
            out.append(self._pad(self._normalise(w, d, "bridge_prev_given_next")))
        return torch.stack(out, dim=1)

    # ------------------------------------------------------------------ model-induced steps
    def model_induced_next_step(self, model_logits: torch.Tensor, x_t: torch.Tensor, n) -> torch.Tensor:
        """
        sum_j p_theta(x_N = j | x_n) P(x_{n+1} = . | x_n, x_N = j), n in [0, N-1].

        Endpoints j that the reference cannot reach from x_n have no bridge and
        are excluded; the remaining mixture is renormalised.
        """
        B = x_t.shape[0]
        n = self._index(n, B, 0, self.N - 1, "model_induced_next_step")
        p_end = torch.softmax(self.masked_logits(model_logits).double(), dim=-1)
        out = []
        for d, k in enumerate(self._kernels):
            S_d = self.cardinalities[d]
            p = p_end[:, d, :S_d]
            reach = k.row("to_end", n, x_t[:, d])                       # Q_{n,N}[x_n, j]
            ok = reach > 0
            ratio = torch.where(ok, p / torch.where(ok, reach, torch.ones_like(reach)), torch.zeros_like(p))
            w = k.row("step", n, x_t[:, d]) * k.matvec("to_end", n + 1, ratio)
            out.append(self._pad(self._normalise(w, d, "model_induced_next_step")))
        return torch.stack(out, dim=1)

    def model_induced_prev_step(self, model_logits: torch.Tensor, x_t: torch.Tensor, n) -> torch.Tensor:
        """sum_i p_theta(x_0 = i | x_n) P(x_{n-1} = . | x_0 = i, x_n), n in [1, N]."""
        B = x_t.shape[0]
        n = self._index(n, B, 1, self.N, "model_induced_prev_step")
        p_start = torch.softmax(self.masked_logits(model_logits).double(), dim=-1)
        out = []
        for d, k in enumerate(self._kernels):
            S_d = self.cardinalities[d]
            p = p_start[:, d, :S_d]
            reach = k.col("from_start", n, x_t[:, d])                   # Q_{0,n}[i, x_n]
            ok = reach > 0
            ratio = torch.where(ok, p / torch.where(ok, reach, torch.ones_like(reach)), torch.zeros_like(p))
            w = k.vecmat("from_start", n - 1, ratio) * k.col("step", n - 1, x_t[:, d])
            out.append(self._pad(self._normalise(w, d, "model_induced_prev_step")))
        return torch.stack(out, dim=1)

    # ------------------------------------------------------------------ sampling
    def sample_from_probs(self, probs: torch.Tensor, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """Draw one category per (row, column). Padded categories have mass exactly 0."""
        B, D, s_max = probs.shape
        if not torch.isfinite(probs).all() or (probs < 0).any():
            raise ValueError("malformed categorical probabilities (non-finite or negative)")
        if (probs * (~self.valid_mask).unsqueeze(0)).sum() != 0:
            raise ValueError("probability mass on padded categories")
        if ((probs.sum(dim=-1) - 1.0).abs() > 1e-6).any():
            raise ValueError("categorical probabilities do not sum to one")
        flat = probs.reshape(-1, s_max)
        return torch.multinomial(flat, num_samples=1, generator=generator).view(B, D)

    def sample_prior(self, n: int, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """Uniform prior endpoint over each column's support; shape (n, D)."""
        cols = [torch.randint(0, c, (n,), device=self.device, generator=generator) for c in self.cardinalities]
        if not cols:
            return torch.zeros((n, 0), dtype=torch.long, device=self.device)
        return torch.stack(cols, dim=1)

    def sample_x_t(self, x_start: torch.Tensor, x_target: torch.Tensor, n,
                   generator: Optional[torch.Generator] = None) -> torch.Tensor:
        return self.sample_from_probs(self.bridge_at_time(x_start, x_target, n), generator=generator)

    def sample_step(self, x_t: torch.Tensor, x_target: torch.Tensor, n,
                    generator: Optional[torch.Generator] = None) -> torch.Tensor:
        return self.sample_from_probs(self.bridge_next_given_prev(x_t, x_target, n), generator=generator)
