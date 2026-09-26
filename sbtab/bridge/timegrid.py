
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal, Optional, Sequence, Tuple

import torch


@dataclass(frozen=True)
class TimeGrid:
    """
    Explicit time grid shared by training and sampling.

    The grid has N = ``num_steps`` intervals and N + 1 points

        t[0] = 0 < t[1] < ... < t[N] = T,        dt[i] = t[i + 1] - t[i] > 0.

    ``grid()`` returns the N + 1 points, ``dt()`` the N step lengths and
    ``total_time()`` the horizon T. A state with index ``i`` lives at ``t[i]``;
    a forward step ``i -> i + 1`` and a backward step ``i + 1 -> i`` both have
    length ``dt[i]``. Every consumer must take the time fed to a network and the
    step length from this one object, so that training and sampling cannot
    disagree about the horizon.

    Schedules
      - "uniform":  dt[i] = horizon / N (``horizon`` defaults to 1).
      - "linear" / "geom": increments interpolate gamma_min..gamma_max. Their sum
        is *not* 1 in general (the 100-step geometric default totals ~0.217914).
        Pass ``horizon`` to rescale the increments so they sum to it exactly.
      - "explicit": the increasing ``points`` are used verbatim (points[0] == 0).

    A unit-time Brownian-bridge interpolation requires T = 1: use
    ``TimeGrid.uniform(N)`` or pass ``horizon=1.0``. DSB variants whose reference
    is deliberately defined on another horizon keep the unnormalized schedules;
    they only need training and sampling to agree on that horizon.

    Legacy API
      ``gammas()`` is ``dt()``. ``times()`` is the *inclusive* cumulative sum
      ``grid()[1:]``, i.e. ``times()[k]`` is the time at the END of interval k.
      It is kept for the DSB solvers; new code should index ``grid()``.
    """
    num_steps: int
    gamma_min: float = 1e-4
    gamma_max: float = 1e-2
    schedule: Literal["linear", "geom", "uniform", "explicit"] = "geom"
    device: Optional[torch.device] = None
    dtype: torch.dtype = torch.float32
    horizon: Optional[float] = None
    points: Optional[Tuple[float, ...]] = None

    def __post_init__(self) -> None:
        if isinstance(self.num_steps, bool) or int(self.num_steps) != self.num_steps or self.num_steps < 1:
            raise ValueError("num_steps must be an integer >= 1")
        if self.horizon is not None and (not math.isfinite(self.horizon) or self.horizon <= 0):
            raise ValueError("horizon must be finite and positive")
        if self.schedule == "explicit":
            if self.points is None:
                raise ValueError("schedule='explicit' requires points")
            pts = tuple(float(p) for p in self.points)
            if len(pts) != self.num_steps + 1:
                raise ValueError("points must have num_steps + 1 entries")
            if not all(math.isfinite(p) for p in pts):
                raise ValueError("points must be finite")
            if pts[0] != 0.0:
                raise ValueError("points[0] must be exactly 0")
            if any(b <= a for a, b in zip(pts[:-1], pts[1:])):
                raise ValueError("points must be strictly increasing")
            if self.horizon is not None and float(self.horizon) != pts[-1]:
                raise ValueError("horizon must equal points[-1] for an explicit grid")
            object.__setattr__(self, "points", pts)
            if self.num_steps < 1:
                raise ValueError("num_steps must be >= 1")
            return

        if self.points is not None:
            raise ValueError("points is only valid with schedule='explicit'")
        if self.schedule not in ("linear", "geom", "uniform"):
            raise ValueError(f"Unknown schedule: {self.schedule}")
        if self.schedule == "uniform":
            if self.num_steps < 1:
                raise ValueError("num_steps must be >= 1")
        elif self.num_steps < 2:
            raise ValueError("num_steps must be >= 2")
        if any(not math.isfinite(g) or g <= 0 for g in (self.gamma_min, self.gamma_max)):
            raise ValueError("gamma_min/gamma_max must be finite and positive")
        if self.gamma_min > self.gamma_max:
            raise ValueError("gamma_min must be <= gamma_max")
        if self.horizon is not None and not self.horizon > 0:
            raise ValueError("horizon must be positive")

    # ------------------------------------------------------------------ builders
    @classmethod
    def uniform(
        cls,
        num_steps: int,
        horizon: float = 1.0,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> "TimeGrid":
        return cls(num_steps=num_steps, schedule="uniform", horizon=float(horizon), device=device, dtype=dtype)

    @classmethod
    def from_points(
        cls,
        points: Sequence[float],
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> "TimeGrid":
        pts = tuple(float(p) for p in points)
        return cls(num_steps=len(pts) - 1, schedule="explicit", points=pts, device=device, dtype=dtype)

    # ------------------------------------------------------------------ float64 core
    def _grid64(self) -> torch.Tensor:
        """Grid points in float64 on CPU; both endpoints are exact."""
        N = self.num_steps
        if self.schedule == "explicit":
            return torch.tensor(self.points, dtype=torch.float64)

        if self.schedule == "uniform":
            T = 1.0 if self.horizon is None else float(self.horizon)
            g = torch.arange(N + 1, dtype=torch.float64) * (T / N)
        else:
            if self.schedule == "linear":
                inc = torch.linspace(self.gamma_min, self.gamma_max, N, dtype=torch.float64)
            else:
                lo = torch.log10(torch.tensor(self.gamma_min, dtype=torch.float64))
                hi = torch.log10(torch.tensor(self.gamma_max, dtype=torch.float64))
                inc = torch.logspace(float(lo), float(hi), N, dtype=torch.float64)
            T = float(inc.sum()) if self.horizon is None else float(self.horizon)
            inc = inc * (T / float(inc.sum()))
            g = torch.cat([torch.zeros(1, dtype=torch.float64), torch.cumsum(inc, dim=0)])

        # Pin the endpoints so accumulated round-off cannot move the horizon.
        g[0] = 0.0
        g[-1] = T
        return g

    # ------------------------------------------------------------------ public API
    def grid(self) -> torch.Tensor:
        """(N + 1,) grid points with grid()[0] == 0 and grid()[N] == T."""
        dev = self.device or torch.device("cpu")
        return self._grid64().to(device=dev, dtype=self.dtype)

    def dt(self) -> torch.Tensor:
        """(N,) positive step lengths, dt[i] = t[i + 1] - t[i]."""
        dev = self.device or torch.device("cpu")
        g = self._grid64()
        return (g[1:] - g[:-1]).to(device=dev, dtype=self.dtype)

    def total_time(self) -> torch.Tensor:
        dev = self.device or torch.device("cpu")
        return self._grid64()[-1].to(device=dev, dtype=self.dtype)

    @property
    def T(self) -> float:
        return float(self._grid64()[-1])

    def time_at(self, idx: torch.Tensor) -> torch.Tensor:
        """Map state indices (0..N) to grid times t[idx]."""
        return self.grid()[idx]

    # ------------------------------------------------------------------ legacy API
    def gammas(self) -> torch.Tensor:
        return self.dt()

    def times(self) -> torch.Tensor:
        """Inclusive cumulative times, i.e. grid()[1:] (end of each interval)."""
        return self.grid()[1:]

    def k_to_t(self, k: torch.Tensor) -> torch.Tensor:
        """
        Map integer interval indices (0..N-1) to the legacy time values times()[k].
        """
        return self.times()[k]

    def remaining_time_from_t(self, t: torch.Tensor) -> torch.Tensor:
        """
        Remaining time until T for time values in [0, T].
        """
        return self.total_time() - t
