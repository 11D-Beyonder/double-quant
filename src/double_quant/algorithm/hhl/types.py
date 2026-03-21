from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class HHLRuntimeParams:
    """
    Runtime HHL circuit parameters produced by a transform strategy.

    These parameters are allocated by the strategy and used to configure
    the HHL circuit construction.
    """

    num_qpe_qubits: int
    norm_const: float
    qpe_evolution_time: float
    ucry_angles: list[float]


class HHLStrategy(Protocol):
    """
    Lifecycle contract for HHL transform strategies.

    A transform strategy defines three phases:
    1. pre_processing: Prepare matrix/vector for quantum processing
    2. allocate_params: Determine circuit parameters (QPE qubits, evolution time, etc.)
    3. post_processing: Convert raw solution back to original space
    """

    def pre_processing(
        self, matrix: np.ndarray, vector: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]: ...

    def allocate_params(self, *, epsilon: float = 1 / 8) -> HHLRuntimeParams: ...

    def post_processing(self, raw_solution: np.ndarray) -> np.ndarray: ...
