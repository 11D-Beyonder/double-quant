"""Transformation strategies for HHL linear-system solving."""

from typing import Protocol
import numpy as np

from .types import HHLRuntimeParams, HHLStrategy


class EigenPredictor(Protocol):
    """Interface for predicting eigenvalue bounds."""

    @property
    def max_abs_eigen(self) -> float: ...

    @property
    def min_abs_eigen(self) -> float: ...


class ExactEigenPredictor(EigenPredictor):
    def __init__(self, matrix) -> None:
        abs_eigens = np.abs(np.linalg.eigvals(matrix))
        self._max_abs_eigen = abs_eigens.max()
        self._min_abs_eigen = abs_eigens.min()

    @property
    def max_abs_eigen(self) -> float:
        return self._max_abs_eigen

    @property
    def min_abs_eigen(self) -> float:
        return self._min_abs_eigen


class EigenBasedStrategy(HHLStrategy):
    """Default HHL transform strategy implementing the SAPO pipeline.
    Zhu, Tianze, et al. "SAPO: Improving the Scalability and Accuracy of
    Quantum Linear Solver for Portfolio Optimization."
    2025 62nd ACM/IEEE Design Automation Conference (DAC). IEEE, 2025.
    """

    def __init__(
        self,
        matrix: np.ndarray,
        vector: np.ndarray,
        eigen_predictor: EigenPredictor | None = None,
    ) -> None:
        self._matrix = matrix
        self._vector = vector
        if eigen_predictor is None:
            eigen_predictor = ExactEigenPredictor(matrix)
        self._eigen_predictor = eigen_predictor
        self._solution_post_scale = None

    def pre_processing(
        self, matrix: np.ndarray, vector: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        vector_norm = float(np.linalg.norm(vector))

        matrix_scale = 0.5 / self._eigen_predictor.max_abs_eigen
        vector_scale = 1.0 / vector_norm

        self._solution_post_scale = matrix_scale / vector_scale

        return matrix, vector

    def allocate_params(
        self, *, epsilon: float = 1 / 8, max_qpe_qubits=8
    ) -> HHLRuntimeParams:
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")

        num_qpe_qubits = min(
            max_qpe_qubits,
            max(
                1,
                2
                + np.ceil(
                    np.log2(
                        self._eigen_predictor.max_abs_eigen
                        / epsilon
                        / self._eigen_predictor.min_abs_eigen
                    )
                ),
            ),
        )
        norm_const = (
            self._eigen_predictor.min_abs_eigen
            / self._eigen_predictor.max_abs_eigen
            / 2
        )

        CLAMP_TOL = 1e-5
        angles = [0.0]
        for i in range(1, 2**num_qpe_qubits):
            phi = i / 2**num_qpe_qubits
            offset = 1 if i >= 2 ** (num_qpe_qubits - 1) else 0
            rotation_value = norm_const * 0.5 / (phi - offset)

            if np.isclose(rotation_value, 1.0, rtol=CLAMP_TOL, atol=CLAMP_TOL):
                angles.append(np.pi)
            elif np.isclose(rotation_value, -1.0, rtol=CLAMP_TOL, atol=CLAMP_TOL):
                angles.append(-np.pi)
            elif -1.0 < rotation_value < 1.0:
                angles.append(2 * np.arcsin(rotation_value))
            else:
                angles.append(0.0)

        return HHLRuntimeParams(
            num_qpe_qubits=num_qpe_qubits,
            norm_const=norm_const,
            qpe_evolution_time=np.pi,
            ucry_angles=angles,
        )

    def post_processing(self, raw_solution: np.ndarray) -> np.ndarray:
        return np.asarray(raw_solution, dtype=float) * self._solution_post_scale
