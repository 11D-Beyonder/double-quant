from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from qiskit.circuit.library import real_amplitudes
from qiskit.primitives import BaseSamplerV2
from qiskit_algorithms import SamplingVQE
from qiskit_algorithms.optimizers import Minimizer, Optimizer

from double_quant.algorithm.qubo._util import (
    build_pauli_operator,
    build_result,
    default_optimizer,
    default_sampler,
    ensure_ising_problem,
)
from double_quant.algorithm.qubo.result import QUBOSolverResult
from double_quant.common import IsingProblem, QUBOProblem


class SamplingVQESolver:
    """Project wrapper around qiskit_algorithms.SamplingVQE."""

    def __init__(
        self,
        sampler: BaseSamplerV2 | None = None,
        ansatz: Any | None = None,
        optimizer: Optimizer | Minimizer | None = None,
        *,
        initial_point: np.ndarray | None = None,
        aggregation: float | Callable[[list[float]], float] | None = None,
        callback: Callable[[int, np.ndarray, float, dict[str, Any]], None] | None = None,
        seed: int | None = None,
    ) -> None:
        self._sampler = sampler or default_sampler(seed)
        self._ansatz = ansatz
        self._optimizer = optimizer or default_optimizer()
        self._initial_point = initial_point
        self._aggregation = aggregation
        self._callback = callback

    def solve(self, problem: QUBOProblem | IsingProblem) -> QUBOSolverResult:
        ising_problem = ensure_ising_problem(problem)
        operator = build_pauli_operator(problem)
        ansatz = self._ansatz or real_amplitudes(ising_problem.num_variables, reps=1)
        algorithm = SamplingVQE(
            sampler=self._sampler,
            ansatz=ansatz,
            optimizer=self._optimizer,
            initial_point=self._initial_point,
            aggregation=self._aggregation,
            callback=self._callback,
        )
        raw_result = algorithm.compute_minimum_eigenvalue(operator=operator)
        return build_result(problem, ising_problem, raw_result)
