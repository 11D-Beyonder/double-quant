from abc import ABC
import numpy as np


class LinearSolverOptimizer(ABC):
    pass


class EigenPredictor:
    @property
    def max_abs_eigen(self) -> float:
        raise NotImplementedError("max_abs_eigen method is not implemented.")

    @property
    def min_abs_eigen(self) -> float:
        raise NotImplementedError("min_abs_eigen method is not implemented.")


class SAPO(LinearSolverOptimizer):
    def __init__(
        self, matrix: np.ndarray, eigen_predictor: EigenPredictor | None = None
    ) -> None:
        self._matrix: np.ndarray = matrix
        self._eigen_predictor: EigenPredictor | None = eigen_predictor
        self._matrix_init_scaling: float | None = None
        self._norm_constant: float | None = None

    @property
    def matrix_init_scaling(self) -> float:
        if self._matrix_init_scaling is None:
            if self._eigen_predictor is None:
                self._matrix_init_scaling = (
                    np.abs(np.linalg.eigvals(self._matrix)).max() * 0.5
                )
            else:
                self._matrix_init_scaling = self._eigen_predictor.max_abs_eigen * 0.5
        assert self._matrix_init_scaling is not None
        return self._matrix_init_scaling

    @property
    def hamiltonian_simulation_time(self) -> float:
        return np.pi

    @property
    def norm_constant(self) -> float:
        if self._norm_constant is None:
            if self._eigen_predictor is None:
                self._norm_constant = np.abs(np.linalg.eigvals(self._matrix)).min()
            else:
                self._norm_constant = self._eigen_predictor.min_abs_eigen
        assert self._norm_constant is not None
        return self._norm_constant
