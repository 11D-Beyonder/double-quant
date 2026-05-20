from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from qiskit import QuantumCircuit
from qiskit.primitives import BaseSamplerV2
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit_algorithms import QAOA
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


class QAOASolver:
    """Project wrapper around qiskit_algorithms.QAOA."""

    def __init__(
        self,
        sampler: BaseSamplerV2 | None = None,
        optimizer: Optimizer | Minimizer | None = None,
        *,
        reps: int = 1,
        initial_state: QuantumCircuit | None = None,
        mixer: QuantumCircuit | BaseOperator | None = None,
        initial_point: np.ndarray | None = None,
        aggregation: float | Callable[[list[float]], float] | None = None,
        callback: Callable[[int, np.ndarray, float, dict[str, Any]], None] | None = None,
        transpiler: Any | None = None,
        transpiler_options: dict[str, Any] | None = None,
        seed: int | None = None,
    ) -> None:
        self._sampler = sampler or default_sampler(seed)
        self._optimizer = optimizer or default_optimizer()
        self._reps = reps
        self._initial_state = initial_state
        self._mixer = mixer
        self._initial_point = initial_point
        self._aggregation = aggregation
        self._callback = callback
        self._transpiler = transpiler
        self._transpiler_options = transpiler_options

    def solve(self, problem: QUBOProblem | IsingProblem) -> QUBOSolverResult:
        ising_problem = ensure_ising_problem(problem)
        operator = build_pauli_operator(problem)
        algorithm_kwargs: dict[str, Any] = {
            "sampler": self._sampler,
            "optimizer": self._optimizer,
            "reps": self._reps,
            "initial_state": self._initial_state,
            "initial_point": self._initial_point,
            "aggregation": self._aggregation,
            "callback": self._callback,
            "transpiler": self._transpiler,
            "transpiler_options": self._transpiler_options,
        }
        if self._mixer is not None:
            algorithm_kwargs["mixer"] = self._mixer
        algorithm = QAOA(**algorithm_kwargs)
        raw_result = algorithm.compute_minimum_eigenvalue(operator=operator)
        return build_result(problem, ising_problem, raw_result)
