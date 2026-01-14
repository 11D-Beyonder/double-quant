import numpy as np


class EigenPredictor:
    @property
    def max_abs_eigen(self) -> float:
        raise NotImplementedError("max_abs_eigen method is not implemented.")

    @property
    def min_abs_eigen(self) -> float:
        raise NotImplementedError("min_abs_eigen method is not implemented.")


class SAPO:
    def __init__(
        self, matrix: np.ndarray, eigen_predictor: EigenPredictor | None = None
    ) -> None:
        self._matrix: np.ndarray = matrix
        self._eigen_predictor: EigenPredictor | None = eigen_predictor
        self._min_abs_eigen: float | None = None
        self._max_abs_eigen: float | None = None
        self._matrix_init_scaling: float | None = None
        self._norm_constant: float | None = None

    @property
    def min_abs_eigen(self) -> float:
        if self._min_abs_eigen is None:
            if self._eigen_predictor is None:
                self._min_abs_eigen = np.abs(np.linalg.eigvals(self._matrix)).min()
            else:
                self._min_abs_eigen = self._eigen_predictor.min_abs_eigen
        assert self._min_abs_eigen is not None
        return self._min_abs_eigen

    @property
    def max_abs_eigen(self) -> float:
        if self._max_abs_eigen is None:
            if self._eigen_predictor is None:
                self._min_abs_eigen = np.abs(np.linalg.eigvals(self._matrix)).max()
            else:
                self._min_abs_eigen = self._eigen_predictor.max_abs_eigen
        assert self._max_abs_eigen is not None
        return self._max_abs_eigen

    @property
    def qpe_qubit_num(self, epsilon=1 / 8) -> int:
        return 2 + np.ceil(np.log2(self.max_abs_eigen / epsilon / self.min_abs_eigen))

    @property
    def matrix_init_scaling(self) -> float:
        if self._matrix_init_scaling is None:
            self._matrix_init_scaling = self.max_abs_eigen / 2
        return self._matrix_init_scaling

    @property
    def hamiltonian_simulation_time(self) -> float:
        return np.pi

    @property
    def norm_constant(self) -> float:
        return self.min_abs_eigen
