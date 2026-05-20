from dataclasses import dataclass
from typing import Self
import numpy as np


class LinearSystem:
    """
    Represents a linear system Ax = b with scaling for quantum algorithms.

    This class handles the normalization and scaling required for quantum linear
    system solvers like HHL (Harrow-Hassidim-Lloyd). It automatically scales the
    matrix and vector to satisfy quantum algorithm requirements:
    - Matrix eigenvalues should be in a suitable range
    - Vector should be normalized

    Attributes:
        matrix: The scaled coefficient matrix A
        vector: The scaled right-hand side vector b
        solution_scaling: Factor to scale the quantum solution back to original scale

    Examples:
        Basic usage:
        >>> A = np.array([[2, 1], [1, 2]])
        >>> b = np.array([3, 3])
        >>> system = LinearSystem(A, b)
        >>> system.matrix  # Scaled matrix
        >>> system.vector  # Normalized vector

        Custom matrix scaling:
        >>> system = LinearSystem(A, b, matrix_init_scaling=0.5)

        Generate random system for testing:
        >>> system = LinearSystem.random_for_hhl(n=4)
    """

    def __init__(
        self,
        matrix: np.ndarray,
        vector: np.ndarray,
        matrix_init_scaling: float | np.floating = 1.0,
        vector_init_scaling: float | np.floating = 1.0,
    ) -> None:
        """
        Initialize a linear system with scaling.

        Args:
            matrix: Coefficient matrix A of shape (n, n). Should be square and
                    symmetric for HHL algorithm.
            vector: Right-hand side vector b of shape (n,).
            matrix_init_scaling: Optional custom scaling factor for the matrix.
                                 If None, uses 1/trace(A) as the scaling factor.

        Raises:
            ValueError: If matrix is not square or dimensions are incompatible.
        """
        # Validate inputs
        if matrix.ndim != 2:
            raise ValueError(f"Matrix must be 2-dimensional, got shape {matrix.shape}")
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError(
                f"Matrix must be square, got shape {matrix.shape[0]}x{matrix.shape[1]}"
            )
        if vector.ndim != 1:
            raise ValueError(f"Vector must be 1-dimensional, got shape {vector.shape}")
        if matrix.shape[0] != vector.shape[0]:
            raise ValueError(
                f"Matrix and vector dimensions incompatible: "
                f"matrix is {matrix.shape[0]}x{matrix.shape[0]}, "
                f"vector has length {vector.shape[0]}"
            )

        # Check if matrix is symmetric (required for HHL)
        if not np.allclose(matrix, matrix.T):
            import warnings

            warnings.warn(
                "Matrix is not symmetric. HHL algorithm requires symmetric (Hermitian) matrices. "
                "Consider symmetrizing: A_sym = (A + A.T) / 2",
                UserWarning,
                stacklevel=2,
            )

        # Scale matrix and vector
        self._matrix_init_scaling = matrix_init_scaling
        self._matrix = matrix * self._matrix_init_scaling
        self._vector_init_scaling = vector_init_scaling
        self._vector = vector * self._vector_init_scaling

    @property
    def matrix(self) -> np.ndarray:
        """Get the scaled coefficient matrix."""
        return self._matrix

    @property
    def vector(self) -> np.ndarray:
        """Get the normalized right-hand side vector."""
        return self._vector

    @property
    def solution_scaling(self):
        """Get the scaling factor to convert solution to original scale."""
        return self._matrix_init_scaling / self._vector_init_scaling

    @property
    def dimension(self) -> int:
        """Get the dimension of the linear system."""
        return self._matrix.shape[0]

    @classmethod
    def random_for_hhl(cls, n: int, rng: np.random.Generator | None = None) -> Self:
        """
        Generate a random linear system suitable for HHL algorithm.

        Creates a random system that satisfies HHL requirements:
        - Matrix is symmetric (real Hermitian: A = A^T)
        - Vector is normalized (||b|| = 1 after scaling)

        Args:
            n: Dimension of the linear system (matrix is n×n, vector is n)
            rng: Optional random number generator for reproducibility

        Returns:
            A LinearSystem with a random symmetric matrix and normalized vector.

        Examples:
            >>> system = LinearSystem.random_for_hhl(4)
            >>> system.dimension
            4
            >>> # Verify matrix is symmetric
            >>> np.allclose(system.matrix, system.matrix.T)
            True

            Reproducible generation:
            >>> rng = np.random.default_rng(seed=42)
            >>> system1 = LinearSystem.random_for_hhl(4, rng=rng)
            >>> rng = np.random.default_rng(seed=42)
            >>> system2 = LinearSystem.random_for_hhl(4, rng=rng)
            >>> np.allclose(system1.matrix, system2.matrix)
            True
        """
        if rng is None:
            rng = np.random.default_rng()

        # Generate random symmetric matrix: S = (A + A^T) / 2
        A = rng.random((n, n))
        matrix = (A + A.T) / 2

        # Generate random unit vector
        b = rng.random(n)
        vector = b

        return cls(matrix, vector)

    def __repr__(self) -> str:
        return (
            f"LinearSystem(dimension={self.dimension}, "
            f"matrix_scaling={self._matrix_init_scaling:.6f}, "
            f"vector_scaling={self._vector_init_scaling:.6f})"
        )


@dataclass(frozen=True, slots=True)
class QUBOProblem:
    """Represents a QUBO objective of the form x^T Q x + c."""

    quadratic_matrix: np.ndarray
    constant: float = 0.0
    variable_names: list[str] | None = None

    def __post_init__(self) -> None:
        matrix = np.asarray(self.quadratic_matrix, dtype=float)
        if matrix.ndim != 2:
            raise ValueError(
                f"quadratic_matrix must be 2-dimensional, got shape {matrix.shape}"
            )
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError(
                "quadratic_matrix must be square, got "
                f"shape {matrix.shape[0]}x{matrix.shape[1]}"
            )
        names = self._normalize_variable_names(matrix.shape[0], self.variable_names)
        object.__setattr__(self, "quadratic_matrix", matrix)
        object.__setattr__(self, "constant", float(self.constant))
        object.__setattr__(self, "variable_names", names)

    @staticmethod
    def _normalize_variable_names(
        dimension: int, variable_names: list[str] | None
    ) -> list[str]:
        if variable_names is None:
            return [f"x_{i}" for i in range(dimension)]
        if len(variable_names) != dimension:
            raise ValueError(
                "variable_names length mismatch: "
                f"expected {dimension}, got {len(variable_names)}"
            )
        return list(variable_names)

    @property
    def num_variables(self) -> int:
        return int(self.quadratic_matrix.shape[0])

    def evaluate(self, bits: np.ndarray | list[int]) -> float:
        bit_array = np.asarray(bits, dtype=int)
        if bit_array.ndim != 1:
            raise ValueError(f"bits must be 1-dimensional, got shape {bit_array.shape}")
        if bit_array.shape[0] != self.num_variables:
            raise ValueError(
                "bits length mismatch: "
                f"expected {self.num_variables}, got {bit_array.shape[0]}"
            )
        if not np.isin(bit_array, (0, 1)).all():
            raise ValueError("bits must contain only 0 or 1")
        value = bit_array @ self.quadratic_matrix @ bit_array
        return float(value + self.constant)


@dataclass(frozen=True, slots=True)
class IsingProblem:
    """Represents an Ising objective of the form h·s + sum J_ij s_i s_j + c."""

    linear_bias: np.ndarray
    quadratic_matrix: np.ndarray
    constant: float = 0.0
    variable_names: list[str] | None = None

    def __post_init__(self) -> None:
        linear = np.asarray(self.linear_bias, dtype=float)
        quadratic = np.asarray(self.quadratic_matrix, dtype=float)
        if linear.ndim != 1:
            raise ValueError(
                f"linear_bias must be 1-dimensional, got shape {linear.shape}"
            )
        if quadratic.ndim != 2:
            raise ValueError(
                f"quadratic_matrix must be 2-dimensional, got shape {quadratic.shape}"
            )
        if quadratic.shape[0] != quadratic.shape[1]:
            raise ValueError(
                "quadratic_matrix must be square, got "
                f"shape {quadratic.shape[0]}x{quadratic.shape[1]}"
            )
        if quadratic.shape[0] != linear.shape[0]:
            raise ValueError(
                "linear_bias and quadratic_matrix size mismatch: "
                f"len(linear_bias)={linear.shape[0]}, "
                f"quadratic_matrix={quadratic.shape}"
            )

        constant = float(self.constant)
        names = QUBOProblem._normalize_variable_names(linear.shape[0], self.variable_names)
        diagonal = np.diag(quadratic).copy()
        constant += float(diagonal.sum())
        symmetric = 0.5 * (quadratic + quadratic.T)
        np.fill_diagonal(symmetric, 0.0)

        object.__setattr__(self, "linear_bias", linear)
        object.__setattr__(self, "quadratic_matrix", symmetric)
        object.__setattr__(self, "constant", constant)
        object.__setattr__(self, "variable_names", names)

    @property
    def num_variables(self) -> int:
        return int(self.linear_bias.shape[0])

    def evaluate(self, spins: np.ndarray | list[int]) -> float:
        spin_array = np.asarray(spins, dtype=int)
        if spin_array.ndim != 1:
            raise ValueError(
                f"spins must be 1-dimensional, got shape {spin_array.shape}"
            )
        if spin_array.shape[0] != self.num_variables:
            raise ValueError(
                "spins length mismatch: "
                f"expected {self.num_variables}, got {spin_array.shape[0]}"
            )
        if not np.isin(spin_array, (-1, 1)).all():
            raise ValueError("spins must contain only -1 or 1")
        pairwise = np.triu(self.quadratic_matrix, k=1)
        energy = self.linear_bias @ spin_array + spin_array @ pairwise @ spin_array
        return float(energy + self.constant)
