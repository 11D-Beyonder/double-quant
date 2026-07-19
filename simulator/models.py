from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np


class SimulationBackend(StrEnum):
    """Supported simulation strategies."""

    STATEVECTOR_CPU = "statevector_cpu"
    TENSOR_NETWORK = "tensor_network"
    NOISY_TENSOR_NETWORK = "noisy_tensor_network"
    SHOT_BASED = "shot_based"


@dataclass(frozen=True, slots=True)
class SimulationResult:
    """Normalized output from a simulator run."""

    backend: SimulationBackend
    num_qubits: int
    counts: dict[str, int] | None = None
    probabilities: dict[str, float] | None = None
    statevector: np.ndarray | None = None
    shots: int | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.num_qubits < 1:
            raise ValueError("num_qubits must be positive")
        if self.counts is not None:
            counts = {str(key): int(value) for key, value in self.counts.items()}
            if any(value < 0 for value in counts.values()):
                raise ValueError("counts cannot contain negative values")
            object.__setattr__(self, "counts", counts)
        if self.probabilities is not None:
            probabilities = {
                str(key): float(value) for key, value in self.probabilities.items()
            }
            if any(value < 0.0 for value in probabilities.values()):
                raise ValueError("probabilities cannot contain negative values")
            object.__setattr__(self, "probabilities", probabilities)
        if self.statevector is not None:
            statevector = np.asarray(self.statevector, dtype=complex)
            if statevector.ndim != 1:
                raise ValueError("statevector must be 1-dimensional")
            object.__setattr__(self, "statevector", statevector)
        if self.shots is not None and self.shots < 1:
            raise ValueError("shots must be positive")
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True, slots=True)
class ComplexityReport:
    """Gate-count and circuit-depth summary."""

    num_qubits: int
    depth: int
    gate_count: int
    operations: dict[str, int]
    two_qubit_gate_count: int
    backend: SimulationBackend | None = None
    optimization_level: int | None = None


@dataclass(frozen=True, slots=True)
class PrecisionReport:
    """Accuracy report for noiseless/noisy simulation and financial objectives."""

    noiseless_fidelity: float | None
    noisy_fidelity: float | None
    total_variation_distance: float | None
    success_probability: float | None
    objective_error: float | None
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CapacityReport:
    """Whether the requested simulator path satisfies the 20+ qubit target."""

    backend: SimulationBackend
    requested_qubits: int
    supported: bool
    minimum_required_qubits: int = 20
    reason: str = ""
