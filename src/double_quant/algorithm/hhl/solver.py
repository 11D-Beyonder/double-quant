from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol, final

import numpy as np
import scipy as sp
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import StatePreparation, UCRYGate, phase_estimation
from qiskit.quantum_info import Statevector

from .sapo import SAPO


@dataclass(frozen=True)
class PreparedSystem:
    """
    Prepared linear system and strategy-owned context.

    This dataclass holds the preprocessed matrix, vector, and any
    strategy-specific context needed for the HHL algorithm.
    """

    matrix: np.ndarray
    vector: np.ndarray
    ctx: Any


class RotationSpec(Protocol):
    """
    Protocol for providing controlled-rotation angles in HHL circuits.

    Different rotation strategies implement this protocol to convert
    phase estimates into rotation angles for amplitude loading.
    """

    def to_angles(self, num_phase_qubits: int) -> list[float]: ...


@dataclass(frozen=True)
class HHLRuntimeParams:
    """
    Runtime HHL circuit parameters produced by a transform strategy.

    These parameters are allocated by the strategy and used to configure
    the HHL circuit construction.
    """

    num_qpe_qubits: int
    norm_const: float
    qpe_evolution_time: float
    controlled_rotation: RotationSpec


class HHLStrategy(Protocol):
    """
    Lifecycle contract for HHL transform strategies.

    A transform strategy defines three phases:
    1. pre_processing: Prepare matrix/vector for quantum processing
    2. allocate_params: Determine circuit parameters (QPE qubits, evolution time, etc.)
    3. post_processing: Convert raw solution back to original space
    """

    def pre_processing(
        self, matrix: np.ndarray, vector: np.ndarray
    ) -> PreparedSystem: ...

    def allocate_params(
        self, prepared: PreparedSystem, *, epsilon: float = 1 / 8
    ) -> HHLRuntimeParams: ...

    def post_processing(self, raw_solution: np.ndarray, *, ctx) -> np.ndarray: ...


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


def _construct_circuit(
    prepared,
    params,
    *,
    max_qpe_qubits: int = 8,
) -> _HhlCircuit:
    """
    Construct the full HHL quantum circuit.

    The HHL circuit consists of four main components:
    1. State Preparation: Prepare the input vector |b> as a quantum state
    2. QPE (Quantum Phase Estimation): Estimate eigenvalues of the matrix
    3. Controlled Rotation: Rotate ancilla qubit based on eigenvalue estimates
    4. Inverse QPE: Uncompute the phase estimation

    Args:
        params: Runtime parameters including QPE qubits, evolution time, rotation spec
        prepared: Prepared linear system with matrix, vector, and context
        max_qpe_qubits: Maximum number of QPE phase qubits (default 8)

    Returns:
        _HhlCircuit: The complete HHL quantum circuit
    """
    num_vector_qubits = int(np.log2(prepared.matrix.shape[0]))

    num_phase_qubits = min(params.qpe_qubits, max_qpe_qubits)

    vector_reg = QuantumRegister(num_vector_qubits, name="vector")
    phase_reg = QuantumRegister(num_phase_qubits, name="phase")
    flag_reg = QuantumRegister(1, name="flag")

    vector_circuit = QuantumCircuit(vector_reg.size, name="State Preparation")
    vector_circuit.append(
        StatePreparation(prepared.vector.tolist()), vector_circuit.qubits
    )

    matrix_circuit = QuantumCircuit(vector_reg.size, name="U")
    matrix_circuit.unitary(
        sp.linalg.expm(1j * prepared.matrix * params.evolution_time),
        matrix_circuit.qubits,
    )

    qpe_circuit = phase_estimation(num_phase_qubits, matrix_circuit)

    ucry_circuit = QuantumCircuit(num_phase_qubits + 1)
    angles = params.rotation_spec.to_angles(num_phase_qubits)
    ucry_circuit.compose(UCRYGate(angles), inplace=True)

    hhl_circuit = _HhlCircuit(
        vector_reg,
        phase_reg,
        flag_reg,
        norm_const=params.norm_const,
        name="HHL",
    )
    hhl_circuit.append(vector_circuit, vector_reg[:])
    hhl_circuit.append(qpe_circuit, phase_reg[:] + vector_reg[:])
    hhl_circuit.append(ucry_circuit, flag_reg[:] + phase_reg[::-1])
    hhl_circuit.append(qpe_circuit.inverse(), phase_reg[:] + vector_reg[:])
    return hhl_circuit


def _extract_transformed_solution_by_statevector(
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
    success_prob = float(states.probabilities_dict(circuit.flag_qubits).get("1", 0.0))

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
            + np.binary_repr(i, circuit.num_vector_qubits)
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
        transform_strategy: HHLStrategy | Literal["sapo", "qiskit"] = "sapo",
        epsilon: float = 1 / 8,
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
        if isinstance(transform_strategy, str):
            if transform_strategy != "sapo":
                raise ValueError(
                    "method is only meaningful for default strategy resolution; "
                    "leave method='sapo' when passing transform_strategy"
                )
            else:
                transform_strategy = SAPO()
        prepared = transform_strategy.pre_processing(matrix, vector)
        params = transform_strategy.allocate_params(prepared, epsilon=epsilon)

        circuit = _construct_circuit(prepared, params)
        raw_solution = _extract_transformed_solution_by_statevector(circuit)
        return transform_strategy.post_processing(raw_solution, ctx=prepared.ctx)
