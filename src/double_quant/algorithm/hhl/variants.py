"""Transformation strategies for HHL linear-system solving."""

from typing import final
import numpy as np
import scipy as sp
from qiskit.circuit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import StatePreparation, UCRYGate, phase_estimation
from qiskit.quantum_info import Statevector


class HHLStrategy:
    def solve(self) -> np.ndarray:
        raise NotImplementedError


@final
class _HhlCircuit(QuantumCircuit):
    """
    Custom quantum circuit for HHL algorithm.

    Registers:
        - vector_reg: Qubits for encoding the solution vector
        - phase_reg: Qubits for quantum phase estimation
        - flag_reg: Single ancilla qubit for success flag
    """

    def __init__(
        self,
        vector_reg: QuantumRegister,
        phase_reg: QuantumRegister,
        flag_reg: QuantumRegister,
        norm_const: float,
        **kwargs,
    ):
        """
        Initialize HHL circuit with quantum registers.

        Args:
            vector_reg: Quantum register for solution vector encoding
            phase_reg: Quantum register for phase estimation
            flag_reg: Single-qubit register for success flag
            norm_const: Normalization constant for solution extraction
        """
        assert flag_reg.size == 1
        self._norm_const = norm_const
        super().__init__(vector_reg, phase_reg, flag_reg, **kwargs)

    @property
    def num_vector_qubits(self) -> int:
        """Number of qubits in the vector register."""
        return self.qregs[0].size

    @property
    def num_phase_qubits(self) -> int:
        """Number of qubits in the phase register (QPE precision)."""
        return self.qregs[1].size

    @property
    def flag_qubits(self) -> list[int]:
        """Qubit indices of the flag register."""
        return list(
            range(
                self.num_vector_qubits + self.num_phase_qubits,
                self.num_qubits,
            )
        )

    @property
    def norm_const(self) -> float:
        """Normalization constant used in solution extraction."""
        return self._norm_const


class EigenPredictor:
    """Interface for predicting eigenvalue bounds."""

    @property
    def max_abs_eigen(self) -> float: ...

    @property
    def min_abs_eigen(self) -> float: ...


class ExactEigenPredictor(EigenPredictor):
    def __init__(self, matrix: np.ndarray) -> None:
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
        epsilon: float = 1 / 8,
        max_qpe_qubits: int = 10,
    ) -> None:
        self._matrix = matrix
        self._vector = vector
        if eigen_predictor is None:
            eigen_predictor = ExactEigenPredictor(matrix)
        self._eigen_predictor = eigen_predictor
        self._max_qpe_qubits = max_qpe_qubits
        self._epsilon = epsilon
        self._solution_post_scale = None

    def solve(self) -> np.ndarray:
        preprocessed_system = self._pre_scaling()
        circuit = self._construct_circuit(*preprocessed_system)
        raw_solution = self._extract_solution_by_statevector(circuit)
        return self._restore_scaling(raw_solution)

    def build_circuit(self):
        """Build the SAPO-style HHL circuit after matrix/vector pre-scaling."""
        return self._construct_circuit(*self._pre_scaling())

    def _pre_scaling(self):
        vector_norm = float(np.linalg.norm(self._vector))

        matrix_scale = 0.5 / self._eigen_predictor.max_abs_eigen
        vector_scale = 1.0 / vector_norm

        self._solution_post_scale = matrix_scale / vector_scale

        return self._matrix * matrix_scale, self._vector * vector_scale

    def _construct_circuit(self, matrix: np.ndarray, vector: np.ndarray):
        num_vector_qubits = int(np.log2(matrix.shape[0]))
        num_phase_qubits = min(
            self._max_qpe_qubits,
            max(
                1,
                2
                + int(
                    np.ceil(
                        np.log2(
                            self._eigen_predictor.max_abs_eigen
                            / self._epsilon
                            / self._eigen_predictor.min_abs_eigen
                        )
                    )
                ),
            ),
        )
        vector_reg = QuantumRegister(num_vector_qubits, name="vector")
        phase_reg = QuantumRegister(num_phase_qubits, name="phase")
        flag_reg = QuantumRegister(1, name="flag")

        vector_circuit = QuantumCircuit(vector_reg.size, name="State Preparation")
        vector_circuit.append(StatePreparation(vector.tolist()), vector_circuit.qubits)

        matrix_circuit = QuantumCircuit(vector_reg.size, name="U")
        matrix_circuit.unitary(
            sp.linalg.expm(1j * matrix * np.pi),
            matrix_circuit.qubits,
        )
        qpe_circuit = phase_estimation(num_phase_qubits, matrix_circuit)

        norm_const = (
            self._eigen_predictor.min_abs_eigen
            / self._eigen_predictor.max_abs_eigen
            / 2
        )
        CLAMP_TOL = 1e-5
        angles = [0.0]
        for i in range(1, 2**num_phase_qubits):
            phi = i / 2**num_phase_qubits
            offset = 1 if i >= 2 ** (num_phase_qubits - 1) else 0
            rotation_value = norm_const * 0.5 / (phi - offset)

            if np.isclose(rotation_value, 1.0, rtol=CLAMP_TOL, atol=CLAMP_TOL):
                angles.append(np.pi)
            elif np.isclose(rotation_value, -1.0, rtol=CLAMP_TOL, atol=CLAMP_TOL):
                angles.append(-np.pi)
            elif -1.0 < rotation_value < 1.0:
                angles.append(2 * np.arcsin(rotation_value))
            else:
                angles.append(0.0)
        ucry_circuit = QuantumCircuit(num_phase_qubits + 1)
        ucry_circuit.compose(UCRYGate(angles), inplace=True)

        hhl_circuit = _HhlCircuit(
            vector_reg, phase_reg, flag_reg, norm_const, name="HHL"
        )
        hhl_circuit.append(vector_circuit, vector_reg[:])
        hhl_circuit.append(qpe_circuit, phase_reg[:] + vector_reg[:])
        hhl_circuit.append(ucry_circuit, flag_reg[:] + phase_reg[::-1])
        hhl_circuit.append(qpe_circuit.inverse(), phase_reg[:] + vector_reg[:])
        return hhl_circuit

    def _restore_scaling(self, raw_solution: np.ndarray) -> np.ndarray:
        return np.asarray(raw_solution, dtype=float) * self._solution_post_scale

    def _extract_solution_by_statevector(
        self,
        circuit: _HhlCircuit,
    ) -> np.ndarray:
        """
        Extract the solution vector from HHL circuit using statevector simulation.

        The solution is encoded in amplitudes where the flag qubit is |1>.
        We extract these amplitudes, normalize them, and correct for the
        success probability and normalization constant used in the circuit.

        Args:
            circuit: The HHL quantum circuit to simulate

        Returns:
            np.ndarray: The transformed solution vector (before post-processing)

        Raises:
            ValueError: If success probability is too low or solution has zero norm
        """
        states = Statevector.from_circuit(circuit)
        success_prob = float(
            states.probabilities_dict(circuit.flag_qubits).get("1", 0.0)
        )

        if success_prob < 1e-10:
            raise ValueError(
                f"Success probability too low: {success_prob:.2e}. "
                "The HHL algorithm failed to produce a valid solution. "
                "Try adjusting the configuration or checking matrix conditioning."
            )

        indices = [
            int(
                "1"
                + "0" * circuit.num_phase_qubits
                + np.binary_repr(i, circuit.num_vector_qubits),
                2,
            )
            for i in range(2**circuit.num_vector_qubits)
        ]

        solution_amplitudes = np.real(states.data)[indices]
        solution_norm = float(np.linalg.norm(solution_amplitudes))
        if solution_norm < 1e-10:
            raise ValueError(
                "Extracted solution has zero norm. "
                "This indicates a problem with the HHL circuit construction."
            )

        transformed_solution = (solution_amplitudes / solution_norm) * (
            np.sqrt(success_prob) / circuit.norm_const
        )
        return np.asarray(transformed_solution, dtype=float)
