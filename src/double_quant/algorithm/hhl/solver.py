from typing import Literal
import numpy as np
from .variants import EigenBasedStrategy

class HHLSolver:
    """
    HHL (Harrow-Hassidim-Lloyd) quantum linear system solver.

    This class provides a static method to solve linear systems Ax = b
    using the HHL quantum algorithm. It supports different transform strategies
    for preprocessing, parameter allocation, and post-processing.
    """

    @staticmethod
    def solve(
        matrix: np.ndarray,
        vector: np.ndarray,
        variant: Literal["sapo", "qiskit"] = "sapo",
        **variant_args,
    ) -> np.ndarray:
        """
        Solve the linear system Ax = b using the HHL quantum algorithm.

        The HHL algorithm solves linear systems exponentially faster than
        classical methods for certain matrix types.

        Args:
            matrix: The coefficient matrix A (must be symmetric/Hermitian)
            vector: The right-hand side vector b
            transform_strategy: Preprocessing strategy ('sapo', 'qiskit', or custom)
            epsilon: Target solution error bound (default 1/8)

        Returns:
            np.ndarray: The solution vector x approximating Ax = b

        Raises:
            ValueError: If transform_strategy is invalid or algorithm fails
        """
        if variant != "sapo":
            raise ValueError(
                "method is only meaningful for default strategy resolution currently; "
                "leave method='sapo' when passing transform_strategy"
            )
        else:
            strategy = EigenBasedStrategy(matrix, vector, **variant_args)
        return strategy.solve()