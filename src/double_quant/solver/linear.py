import numpy as np
import scipy as sp
from abc import ABC, abstractmethod


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

    def solve(self, A: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Solve the linear system Ax = b using SciPy's implementation.

        Raises:
            np.linalg.LinAlgError: If the matrix is singular or ill-conditioned.
            ValueError: If the input dimensions are incompatible.
        """
        return sp.linalg.solve(A, b)
