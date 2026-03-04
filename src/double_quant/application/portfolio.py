from __future__ import annotations

import numpy as np

from double_quant.algorithm.hhl import HHLSolver


class PortfolioOptimizer:
    def __init__(
        self,
        expected_returns: np.ndarray,
        covariance: np.ndarray,
        target_return: float,
        assets: list[str] | None = None,
        solver_class: type = HHLSolver,
        **solver_kwargs,
    ) -> None:
        mu = np.asarray(expected_returns, dtype=float)
        sigma = np.asarray(covariance, dtype=float)

        if mu.ndim != 1:
            raise ValueError(f"expected_returns must be 1D, got shape {mu.shape}")
        if sigma.ndim != 2:
            raise ValueError(f"covariance must be 2D, got shape {sigma.shape}")
        if sigma.shape[0] != sigma.shape[1]:
            raise ValueError(
                f"covariance must be square, got shape {sigma.shape[0]}x{sigma.shape[1]}"
            )
        if sigma.shape[0] != mu.shape[0]:
            raise ValueError(
                "expected_returns and covariance size mismatch: "
                f"len(expected_returns)={mu.shape[0]}, covariance={sigma.shape}"
            )
        if not np.isfinite(mu).all():
            raise ValueError("expected_returns contains non-finite values")
        if not np.isfinite(sigma).all():
            raise ValueError("covariance contains non-finite values")
        if not np.isfinite(target_return):
            raise ValueError("target_return must be finite")

        num_assets = mu.shape[0]
        if assets is None:
            assets = [f"asset_{i}" for i in range(num_assets)]
        if len(assets) != num_assets:
            raise ValueError(
                f"assets length mismatch: expected {num_assets}, got {len(assets)}"
            )

        self._mu = mu
        self._sigma = sigma
        self._target_return = target_return
        self._assets = assets
        self._num_assets = num_assets
        self._solver_class = solver_class
        self._solver_kwargs = solver_kwargs

    def _build_black_system(self) -> tuple[np.ndarray, np.ndarray]:
        dim = self._num_assets + 2
        matrix = np.zeros((dim, dim), dtype=float)

        matrix[0, 2:] = self._mu
        matrix[1, 2:] = 1.0
        matrix[2:, 0] = self._mu
        matrix[2:, 1] = 1.0
        matrix[2:, 2:] = self._sigma

        vector = np.zeros(dim, dtype=float)
        vector[0] = self._target_return
        vector[1] = 1.0
        return matrix, vector

    @staticmethod
    def _expand_to_power_of_two(
        matrix: np.ndarray, vector: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        dim = matrix.shape[0]
        if dim & (dim - 1) == 0:
            return matrix, vector

        target_dim = 1 << (dim - 1).bit_length()

        expanded_matrix = np.zeros((target_dim, target_dim), dtype=float)
        expanded_vector = np.zeros(target_dim, dtype=float)

        expanded_matrix[:dim, :dim] = matrix
        expanded_vector[:dim] = vector

        extra_dim = target_dim - dim
        assert extra_dim > 0
        expanded_matrix[dim:, dim:] = np.eye(extra_dim, dtype=float)

        return expanded_matrix, expanded_vector

    def _validate_solution_constraints(
        self, weights: np.ndarray, tol: float = 1e-4
    ) -> None:
        weight_sum = float(np.sum(weights))
        achieved_return = float(weights @ self._mu)

        if abs(weight_sum - 1.0) > tol:
            raise ValueError(
                "Optimized solution violates budget constraint: "
                f"sum(w)={weight_sum:.8f}, expected 1.0"
            )
        if abs(achieved_return - self._target_return) > tol:
            raise ValueError(
                "Optimized solution violates target return constraint: "
                f"w^T mu={achieved_return:.8f}, expected {self._target_return:.8f}"
            )

    def optimize(self) -> dict[str, float]:
        matrix, vector = self._build_black_system()
        matrix, vector = self._expand_to_power_of_two(matrix, vector)

        solution = np.asarray(
            self._solver_class.solve(matrix, vector, **self._solver_kwargs), dtype=float
        )

        start = 2
        end = start + self._num_assets
        weights = solution[start:end]
        self._validate_solution_constraints(weights)

        return {asset: float(weights[i]) for i, asset in enumerate(self._assets)}
