from __future__ import annotations

from typing import Any

from qiskit_algorithms import NumPyMinimumEigensolver

from double_quant.algorithm.qubo._util import (
    build_exact_result,
    build_pauli_operator,
    ensure_ising_problem,
)
from double_quant.algorithm.qubo.result import QUBOSolverResult
from double_quant.common import IsingProblem, QUBOProblem


class NumPyMinimumEigensolverSolver:
    """Exact classical baseline solver for QUBO and Ising problems."""

    def __init__(self, filter_criterion: Any | None = None) -> None:
        self._filter_criterion = filter_criterion

    def solve(self, problem: QUBOProblem | IsingProblem) -> QUBOSolverResult:
        ising_problem = ensure_ising_problem(problem)
        operator = build_pauli_operator(problem)
        algorithm = NumPyMinimumEigensolver(filter_criterion=self._filter_criterion)
        raw_result = algorithm.compute_minimum_eigenvalue(operator=operator)
        return build_exact_result(problem, ising_problem, raw_result)
