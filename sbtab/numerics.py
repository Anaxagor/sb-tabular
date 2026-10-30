"""Numerical failure checks shared by torch models; no data repair or clipping of samples."""
from __future__ import annotations

import math
from typing import Iterable

import torch


class NumericalError(FloatingPointError):
    """A numerical failure with JSON-compatible diagnostic context."""

    def __init__(self, message: str, **details) -> None:
        self.details = details
        context = ", ".join(f"{key}={value}" for key, value in details.items())
        super().__init__(f"{message}" + (f" ({context})" if context else ""))


def require_finite(tensor: torch.Tensor, name: str, **context) -> None:
    """Reject NaN/Inf without changing values or exposing individual observations."""
    values = tensor.detach()
    finite = torch.isfinite(values)
    if bool(finite.all()):
        return
    raise NumericalError(
        f"non-finite {name}", **context, tensor=name, shape=list(values.shape),
        n_nonfinite=int((~finite).sum().item()),
        n_nan=int(torch.isnan(values).sum().item()),
        n_posinf=int(torch.isposinf(values).sum().item()),
        n_neginf=int(torch.isneginf(values).sum().item()),
    )


@torch.no_grad()
def check_gradients(
    parameters: Iterable[torch.Tensor], *, max_norm: float | None = None, **context
) -> torch.Tensor:
    """Validate gradients before an optimizer step, then optionally clip their L2 norm.

    Norms and the clipping ratio use float64 so large finite float32 gradients
    cannot overflow the norm and get silently multiplied by zero. No gradient
    is modified if any gradient or its total norm is non-finite.
    """
    if max_norm is not None and (not math.isfinite(float(max_norm)) or float(max_norm) < 0):
        raise ValueError("max_norm must be finite and nonnegative or None")
    entries = [(index, p.grad) for index, p in enumerate(parameters) if p.grad is not None]
    gradients = [gradient for _, gradient in entries]
    if not gradients:
        return torch.tensor(0.0, dtype=torch.float64)
    values = [g.coalesce().values() if g.is_sparse else g for g in gradients]
    finite = torch.stack([torch.isfinite(g).all() for g in values])
    if not bool(finite.all()):
        index = int((~finite).nonzero()[0].item())
        require_finite(values[index], "gradient", parameter_index=entries[index][0], **context)
    norms = torch.stack([torch.linalg.vector_norm(g, dtype=torch.float64) for g in values])
    total_norm = torch.linalg.vector_norm(norms)
    require_finite(total_norm, "gradient_norm", **context)
    if max_norm is not None:
        scale = (float(max_norm) / (total_norm + 1e-6)).clamp(max=1.0)
        for gradient in gradients:
            gradient.mul_(scale.to(device=gradient.device, dtype=gradient.dtype))
    return total_norm
