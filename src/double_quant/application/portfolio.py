from __future__ import annotations

import numpy as np

from double_quant.algorithm.hhl import HHLSolver
from double_quant.common import util


class PortfolioOptimizer:
    def __init__(
        self,
        expected_returns: np.ndarray,
        covariance: np.ndarray,
        target_return: float,
        assets: list[str] | None = None,
        constraint_scaler: ConstraintScaler
        | tuple[float, float, float]
        | tuple[float, float]
        | list[float]
        | str
        | None = None,
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

        if constraint_scaler is None:
            constraint_scaler = ConstraintScaler()
        elif isinstance(constraint_scaler, str):
            constraint_scaler = ConstraintScaler.from_pickle(constraint_scaler)
        elif not isinstance(constraint_scaler, ConstraintScaler):
            constraint_scaler = ConstraintScaler(constraint_scaler)

        self._mu = mu
        self._sigma = sigma
        self._target_return = target_return
        self._assets = assets
        self._num_assets = num_assets
        self._constraint_scaler = constraint_scaler
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
            util.warning(
                "Optimized solution violates budget constraint: "
                f"sum(w)={weight_sum:.8f}, expected 1.0"
            )
        if abs(achieved_return - self._target_return) > tol:
            util.warning(
                "Optimized solution violates target return constraint: "
                f"w^T mu={achieved_return:.8f}, expected {self._target_return:.8f}"
            )

    def optimize(self) -> dict[str, float]:
        matrix, vector = self._build_black_system()
        matrix, vector = self._expand_to_power_of_two(matrix, vector)
        # the answer would not change after constraint scaling
        matrix, vector = self._constraint_scaler.scale(matrix, vector, self._num_assets)

        solution = np.asarray(
            self._solver_class.solve(matrix, vector, **self._solver_kwargs), dtype=float
        )

        weights = solution[2 : 2 + self._num_assets]
        self._validate_solution_constraints(weights)

        return {asset: float(weights[i]) for i, asset in enumerate(self._assets)}


class ConstraintScaler:
    # TODO: implement constraint scaler
    # Train from historical market data
    # Use the existed factors
    def __init__(
        self,
        factors: tuple[float, float, float]
        | tuple[float, float]
        | list[float]
        | None = None,
    ):
        self._factors = factors

    @staticmethod
    def from_pickle(path: str) -> ConstraintScaler: ...

    def scale(
        self, matrix: np.ndarray, vector: np.ndarray, num_assets: int
    ) -> tuple[np.ndarray, np.ndarray]:
        base_matrix = np.asarray(matrix, dtype=float)
        base_vector = np.asarray(vector, dtype=float)
        if self._factors is None:
            return base_matrix, base_vector
        else:
            # TODO: using factor to scale the matrix and vector
            raise NotImplementedError

        # s1, s2, s_star = self._factors
        # scaled_matrix = base_matrix
        # scaled_vector = base_vector

        # tail_dim = self._detect_tail_identity_size(base_matrix)
        # num_assets = dim - 2 - tail_dim
        # if num_assets <= 0:
        #     return scaled_matrix, scaled_vector

        # w_start = 2
        # w_end = 2 + num_assets

        # scaled_matrix[0, w_start:w_end] *= s1
        # scaled_matrix[w_start:w_end, 0] *= s1
        # scaled_matrix[1, w_start:w_end] *= s2
        # scaled_matrix[w_start:w_end, 1] *= s2

        # scaled_vector[0] *= s1
        # scaled_vector[1] *= s2

        # if tail_dim > 0:
        #     split = dim - tail_dim
        #     scaled_matrix[split:, split:] *= s_star

        # return scaled_matrix, scaled_vector
