from abc import ABC, abstractmethod
from typing import final

import numpy as np
import scipy as sp
from qiskit import (
    QuantumCircuit,
    QuantumRegister,
)
from qiskit.circuit.library import (
    PhaseEstimation,
    StatePreparation,
)
from typing_extensions import override

from double_quant.optimizer import SAPO, LinearSolverOptimizer


class LinearSolver(ABC):
    """
    Abstract base class for linear system solvers.

    Defines the interface for solving linear equations of the form Ax = b,
    where A is a square matrix and b is a vector.
    """

    @abstractmethod
    def solve(self, A: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Solve the linear system Ax = b.

        Args:
            A: Coefficient matrix of shape (n, n). Must be square and non-singular.
            b: Right-hand side vector of shape (n,).

        Returns:
            Solution vector x of shape (n,) that satisfies Ax = b.
        """
        raise NotImplementedError("Subclasses must implement the solve method")


class NumPyLinearSolver(LinearSolver):
    """
    Linear solver implementation using NumPy's linalg.solve function.

    Provides efficient solutions for dense linear systems using NumPy's
    optimized LAPACK backend.
    """

    @override
    def solve(self, A: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Solve the linear system Ax = b using NumPy's implementation.

        Raises:
            np.linalg.LinAlgError: If the matrix is singular or ill-conditioned.
            ValueError: If the input dimensions are incompatible.
        """
        return np.linalg.solve(A, b)


class SciPyLinearSolver(LinearSolver):
    """
    Linear solver implementation using SciPy's linalg.solve function.

    Provides solutions for linear systems using SciPy's implementation,
    which may offer additional features and performance optimizations
    compared to NumPy for certain problem types.
    """

    @override
    def solve(self, A: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Solve the linear system Ax = b using SciPy's implementation.

        Raises:
            np.linalg.LinAlgError: If the matrix is singular or ill-conditioned.
            ValueError: If the input dimensions are incompatible.
        """
        return sp.linalg.solve(A, b)


class HhlCircuit(QuantumCircuit):
    @property
    def phase_qubit_num(self) -> int:
        raise NotImplementedError


class QuantumLinearSolver(LinearSolver):
    @final
    class LinearSystem:
        def __init__(self, matrix: np.ndarray, vector: np.ndarray) -> None:
            # TODO: check both size
            self.matrix = matrix
            self.vector = vector
            self._validate_self()

        def _validate_self(self):
            # TODO: validate a linear system
            ...

    def __init__(self, optimizer: LinearSolverOptimizer | None = None) -> None:
        super().__init__()
        self._optimizer: LinearSolverOptimizer | None = optimizer
        self._hhl_circuit: HhlCircuit | None = None
        self._linear_system: QuantumLinearSolver.LinearSystem | None = None

    @override
    def solve(self, A: np.ndarray, b: np.ndarray) -> np.ndarray:
        self._linear_system = self.LinearSystem(A, b)
        # TODO: construct circuit and then extract the solution.
        circuit = self._construct_circuit()

    def _construct_circuit(self):
        assert self._linear_system is not None, (
            "Linear system must be initialized before constructing circuit"
        )
        assert isinstance(self._optimizer, SAPO)

        vector_reg = QuantumRegister(
            int(np.log2(np.shape(self._linear_system.vector))), name="vector"
        )
        phase_reg = QuantumRegister(self._optimizer.qpe_qubit_num, name="phase")
        flag_reg = QuantumRegister(1, name="flag")

        vector_circuit = QuantumCircuit(vector_reg.size, name="vector")
        vector_circuit.append(StatePreparation(self._linear_system.vector.tolist()))

        matrix_circuit = QuantumCircuit(vector_reg.size, name="U")
        matrix_circuit.unitary(
            sp.linalg.expm(
                1j
                * self._linear_system.matrix
                * self._optimizer.hamiltonian_simulation_time
            ),
            matrix_circuit.qubits,
        )
        qpe_gate = PhaseEstimation(phase_reg.size, matrix_circuit)

        # control rotation

        ans = QuantumCircuit(vector_reg, phase_reg, flag_reg, name="HHL")
        # TODO: assembly circuit
        return ans
