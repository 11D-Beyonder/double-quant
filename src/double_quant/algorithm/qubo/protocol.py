from __future__ import annotations

from typing import Protocol, runtime_checkable

from double_quant.algorithm.qubo.result import QUBOSolverResult
from double_quant.common import IsingProblem, QUBOProblem


@runtime_checkable
class QUBOSolver(Protocol):
    """Protocol for solver implementations that accept QUBO or Ising inputs."""

    def solve(self, problem: QUBOProblem | IsingProblem) -> QUBOSolverResult:
        """Solve the given problem and return a normalized result object."""
        ...
