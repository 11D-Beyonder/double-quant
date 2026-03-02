"""
Optimizers for quantum linear system solvers.

This module provides optimization strategies for quantum algorithms, particularly
the SAPO (Scalable Adiabatic-inspired Phase Optimization) method for HHL.
"""

import numpy as np

from double_quant.common import LinearSystem


class EigenPredictor:
    """
    Interface for predicting eigenvalue bounds without full computation.

    Custom eigenvalue predictors can implement this interface to provide
    estimated bounds for matrix eigenvalues, potentially avoiding expensive
    full eigenvalue computation.

    This is particularly useful for large matrices or when domain knowledge
    provides good eigenvalue estimates (e.g., from matrix structure or previous runs).

    Examples:
        >>> class SimplePredictor(EigenPredictor):
        ...     def __init__(self, min_eigen, max_eigen):
        ...         self._min = min_eigen
        ...         self._max = max_eigen
        ...
        ...     @property
        ...     def min_abs_eigen(self) -> float:
        ...         return self._min
        ...
        ...     @property
        ...     def max_abs_eigen(self) -> float:
        ...         return self._max
        >>>
        >>> predictor = SimplePredictor(0.1, 2.0)
        >>> sapo = SAPO(linear_system, eigen_predictor=predictor)
    """

    @property
    def max_abs_eigen(self) -> float:
        """Maximum absolute eigenvalue of the matrix."""
        raise NotImplementedError("max_abs_eigen method is not implemented.")

    @property
    def min_abs_eigen(self) -> float:
        """Minimum absolute eigenvalue of the matrix."""
        raise NotImplementedError("min_abs_eigen method is not implemented.")


class SAPO:
    """
    Scalable Adiabatic-inspired Phase Optimization for HHL algorithm.

    SAPO optimizes the HHL algorithm by:
    1. Scaling the matrix based on maximum eigenvalue (improves conditioning)
    2. Computing eigenvalue bounds for optimal QPE qubit allocation
    3. Providing normalization constants for amplitude amplification

    The optimizer produces a prepared linear system with optimal scaling and
    provides parameters for efficient quantum circuit construction.

    Attributes:
        prepared_linear_system: The optimally scaled linear system
        min_abs_eigen: Minimum absolute eigenvalue (computed or predicted)
        max_abs_eigen: Maximum absolute eigenvalue (computed or predicted)
        qpe_qubit_num: Recommended number of QPE qubits
        norm_constant: Normalization constant for amplitude amplification

    Examples:
        Basic usage:
        >>> import numpy as np
        >>> from double_quant.optimizer import SAPO
        >>> A = np.array([[2, 1], [1, 2]])
        >>> b = np.array([3, 3])
        >>> sapo = SAPO(A, b)
        >>> sapo.get_qpe_qubit_num()  # Recommended QPE qubits
        >>> prepared = sapo.prepared_linear_system

        Custom epsilon for higher precision:
        >>> sapo = SAPO(A, b)
        >>> num_qubits = sapo.get_qpe_qubit_num(epsilon=1/16)

        Using eigen predictor:
        >>> predictor = MyEigenPredictor()
        >>> sapo = SAPO(A, b, eigen_predictor=predictor)
    """

    def __init__(
        self,
        matrix: np.ndarray,
        vector: np.ndarray,
        eigen_predictor: EigenPredictor | None = None,
    ) -> None:
        """
        Initialize SAPO optimizer for a linear system.

        Args:
            matrix: Coefficient matrix A of shape (n, n). Should be square and
                   symmetric for HHL algorithm.
            vector: Right-hand side vector b of shape (n,).
            eigen_predictor: Optional predictor for eigenvalue bounds.
                            If None, eigenvalues will be computed exactly using
                            np.linalg.eigvals.

        Notes:
            - The optimizer creates a scaled LinearSystem internally
            - Matrix is scaled by 0.5 / max_eigenvalue for optimal conditioning
            - Vector is normalized to unit norm
            - Eigenvalues are computed/predicted only when needed (lazy evaluation)
        """
        self._eigen_predictor: EigenPredictor | None = eigen_predictor

        # Create a new scaled system based on maximum eigenvalue
        # This improves the conditioning for HHL
        max_eigen = (
            np.abs(np.linalg.eigvals(matrix)).max()
            if eigen_predictor is None
            else eigen_predictor.max_abs_eigen
        )

        self._linear_system = LinearSystem(
            matrix,
            vector,
            matrix_init_scaling=0.5 / max_eigen,
            vector_init_scaling=1.0 / np.linalg.norm(vector),
        )

        # Cache for lazy evaluation
        self._min_abs_eigen: float | None = None
        self._max_abs_eigen: float | None = None
        self._norm_constant: float | None = None

    @property
    def prepared_linear_system(self) -> LinearSystem:
        """Get the prepared linear system with optimal scaling."""
        return self._linear_system

    @property
    def min_abs_eigen(self) -> float:
        """
        Get minimum absolute eigenvalue of the prepared system.

        Uses eigenpredictor if provided, otherwise computes exactly.
        Result is cached after first computation.
        """
        if self._min_abs_eigen is None:
            if self._eigen_predictor is None:
                self._min_abs_eigen = np.abs(
                    np.linalg.eigvals(self._linear_system.matrix)
                ).min()
            else:
                self._min_abs_eigen = self._eigen_predictor.min_abs_eigen
        assert self._min_abs_eigen is not None
        return self._min_abs_eigen

    @property
    def max_abs_eigen(self) -> float:
        """
        Get maximum absolute eigenvalue of the prepared system.

        Uses eigenpredictor if provided, otherwise computes exactly.
        Result is cached after first computation.
        """
        if self._max_abs_eigen is None:
            if self._eigen_predictor is None:
                self._max_abs_eigen = np.abs(
                    np.linalg.eigvals(self._linear_system.matrix)
                ).max()
            else:
                self._max_abs_eigen = self._eigen_predictor.max_abs_eigen
        assert self._max_abs_eigen is not None
        return self._max_abs_eigen

    def get_qpe_qubit_num(self, epsilon=1 / 8) -> int:
        """
        Get recommended number of QPE (Quantum Phase Estimation) qubits.

        The formula is: n_qpe = 2 + ceil(log2(λ_max / (epsilon * λ_min)))
        where λ_max and λ_min are max and min absolute eigenvalues.

        More qubits provide better phase resolution, allowing accurate
        estimation of eigenvalues for better conditioned systems.

        Returns:
            Recommended number of QPE qubits
        """
        return 2 + int(
            np.ceil(np.log2(self.max_abs_eigen / (epsilon * self.min_abs_eigen)))
        )

    @property
    def norm_const(self) -> float:
        """
        Get normalization constant for amplitude amplification.

        This constant is used in the controlled rotation gate to achieve
        the desired amplitude amplification effect in HHL.

        Returns:
            Normalization constant (equals min_abs_eigen for SAPO)
        """
        return self.min_abs_eigen

    def __repr__(self) -> str:
        return f"eigen_range=[{self.min_abs_eigen:.6f}, {self.max_abs_eigen:.6f}])"
