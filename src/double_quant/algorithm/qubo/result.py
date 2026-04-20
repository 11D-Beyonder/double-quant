from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class QUBOSolverResult:
    """Normalized solver output with bitstring-oriented accessors."""

    best_bitstring: np.ndarray
    best_objective: float
    best_energy: float
    best_probability: float | None = None
    parameter_values: np.ndarray | None = None
    probabilities: dict[str, float] | None = None
    metadata: dict[str, object] | None = None

    def __post_init__(self) -> None:
        bitstring = np.asarray(self.best_bitstring, dtype=int)
        if bitstring.ndim != 1:
            raise ValueError(
                f"best_bitstring must be 1-dimensional, got shape {bitstring.shape}"
            )
        if not np.isin(bitstring, (0, 1)).all():
            raise ValueError("best_bitstring must contain only 0 or 1")
        object.__setattr__(self, "best_bitstring", bitstring)
        object.__setattr__(self, "best_objective", float(self.best_objective))
        object.__setattr__(self, "best_energy", float(self.best_energy))
        if self.best_probability is not None:
            object.__setattr__(self, "best_probability", float(self.best_probability))
        if self.parameter_values is not None:
            object.__setattr__(
                self, "parameter_values", np.asarray(self.parameter_values, dtype=float)
            )
        if self.probabilities is not None:
            normalized = {str(k): float(v) for k, v in self.probabilities.items()}
            object.__setattr__(self, "probabilities", normalized)
        if self.metadata is not None:
            object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def num_variables(self) -> int:
        return int(self.best_bitstring.shape[0])

