"""Shared runtime boundary implemented by every model-specific adapter."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from sbtab.benchmark.contracts import InputSpec, PreparedTable
from sbtab.benchmark.validation import ContractViolation, validate_input_spec


def _validate_seed(seed: int, field_name: str) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ContractViolation(f"{field_name} must be an integer.")
    if not 0 <= seed < 2**32:
        raise ContractViolation(f"{field_name} must be in the range [0, 2**32).")


@dataclass(frozen=True)
class RunContext:
    """Model-independent controls for one adapter fit on one fold.

    Parameters
    ----------
    run_id:
        Stable label used to correlate logs and artifacts for one benchmark run.
        Shared code and adapters must not branch on its value.
    fold_id:
        Zero-based position of the current fold in the common split sequence.
    seed:
        Non-negative 32-bit seed for native model initialization and training.
    device:
        Native execution-device string, for example ``"cpu"`` or ``"cuda:0"``.
        The adapter interprets it and must fail if its backend cannot use it.
    artifact_dir:
        Fold-specific destination assigned by the runner for checkpoints or
        logs. Constructing the context neither creates nor modifies this path.
    """

    run_id: str
    fold_id: int
    seed: int
    device: str
    artifact_dir: Path

    def __post_init__(self) -> None:
        """Reject ambiguous runtime controls before native model construction."""

        if not isinstance(self.run_id, str) or not self.run_id.strip():
            raise ContractViolation("RunContext.run_id must be a non-empty string.")
        if isinstance(self.fold_id, bool) or not isinstance(self.fold_id, int):
            raise ContractViolation("RunContext.fold_id must be an integer.")
        if self.fold_id < 0:
            raise ContractViolation("RunContext.fold_id must be non-negative.")
        _validate_seed(self.seed, "RunContext.seed")
        if not isinstance(self.device, str) or not self.device.strip():
            raise ContractViolation("RunContext.device must be a non-empty string.")
        if not isinstance(self.artifact_dir, Path):
            raise ContractViolation("RunContext.artifact_dir must be pathlib.Path.")


@runtime_checkable
class ModelAdapter(Protocol):
    """Thin translation between canonical prepared tables and one native API.

    One adapter instance belongs to one codec and fold. ``fit`` receives the
    complete prepared train table, including target. ``sample`` returns all
    modeled columns with the exact same :class:`PreparedSchema` object.
    """

    @property
    def name(self) -> str:
        """Return a stable model-family label used only in artifacts and logs."""

        ...

    @property
    def input_spec(self) -> InputSpec:
        """Return the three approved semantic views required by this model."""

        ...

    def fit(self, train: PreparedTable, context: RunContext) -> None:
        """Fit native model state from one validated prepared train fold."""

        ...

    def sample(self, n: int, seed: int) -> PreparedTable:
        """Generate ``n`` complete prepared rows using one explicit seed."""

        ...


def validate_adapter_definition(adapter: ModelAdapter) -> None:
    """Validate stable adapter metadata without fitting or sampling it."""

    if not isinstance(adapter, ModelAdapter):
        raise ContractViolation(
            "adapter must implement the ModelAdapter runtime protocol."
        )
    if not callable(adapter.fit) or not callable(adapter.sample):
        raise ContractViolation(
            "ModelAdapter.fit and ModelAdapter.sample must be callable methods."
        )
    if not isinstance(adapter.name, str) or not adapter.name.strip():
        raise ContractViolation("ModelAdapter.name must be a non-empty string.")
    validate_input_spec(adapter.input_spec)


def validate_sample_request(n: int, seed: int) -> None:
    """Validate model-independent sampling controls shared by all adapters."""

    if isinstance(n, bool) or not isinstance(n, int):
        raise ContractViolation("sample n must be an integer.")
    if n <= 0:
        raise ContractViolation("sample n must be positive.")
    _validate_seed(seed, "sample seed")
