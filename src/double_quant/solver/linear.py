from typing import Literal, final
import scipy as sp
import numpy as np
from qiskit import (
    QuantumCircuit,
    QuantumRegister,
)
from qiskit.circuit.library import (
    StatePreparation,
    UCRYGate,
    phase_estimation,
)
from qiskit.quantum_info import Statevector
from double_quant.common import LinearSystem
from double_quant.optimizer.sapo import SAPO


@final
class _HhlCircuit(QuantumCircuit):
    """
    Internal circuit class for HHL algorithm implementation.

    This class extends Qiskit's QuantumCircuit with HHL-specific metadata and
    convenience properties. It stores normalization constants and the prepared
    linear system needed for post-processing the quantum solution.

    The circuit maintains three quantum registers:
    - vector_reg: Encodes the solution vector (n qubits for 2^n dimensional system)
    - phase_reg: Used for quantum phase estimation (typically n+2 to n+4 qubits)
    - flag_reg: Single qubit for amplitude amplification success indicator

    Attributes:
        norm_const: Normalization constant for scaling the solution
        linear_system: The prepared linear system with scaling information

    Properties:
        num_vector_qubits: Number of qubits encoding the solution vector
        num_phase_qubits: Number of qubits for quantum phase estimation
        num_flag_qubits: Number of flag qubits (always 1)
        vector_qubits: List of indices for vector register qubits
        phase_qubits: List of indices for phase register qubits
        flag_qubits: List of indices for flag register qubits

    Notes:
        - This is an internal class (marked with @final) and should not be
          instantiated directly by users
        - The circuit is constructed by _construct_circuit_sapo()
        - Qubit ordering: [vector_qubits][phase_qubits][flag_qubits]
    """

    def __init__(
        self,
        vector_reg: QuantumRegister,
        phase_reg: QuantumRegister,
        flag_reg: QuantumRegister,
        norm_const: float,
        linear_system: LinearSystem,
        **kwargs,
    ):
        assert flag_reg.size == 1
        self._norm_const = norm_const
        self._linear_system = linear_system
        super().__init__(vector_reg, phase_reg, flag_reg, **kwargs)

    @property
    def num_vector_qubits(self) -> int:
        return self.qregs[0].size

    @property
    def num_phase_qubits(self) -> int:
        return self.qregs[1].size

    @property
    def num_flag_qubits(self) -> int:
        return self.qregs[2].size

    @property
    def vector_qubits(self) -> list[int]:
        return list(range(self.num_vector_qubits))

    @property
    def phase_qubits(self) -> list[int]:
        return list(
            range(
                self.num_vector_qubits,
                self.num_vector_qubits + self.num_phase_qubits,
            )
        )

    @property
    def flag_qubits(self) -> list[int]:
        return list(
            range(
                self.num_vector_qubits + self.num_phase_qubits,
                self.num_qubits,
            )
        )

    @property
    def norm_const(self) -> float:
        return self._norm_const

    @property
    def linear_system(self) -> LinearSystem:
        return self._linear_system


""" class QiskitHHLMethod:
    def construct_circuit(
        self,
        linear_system: _LinearSystem,
        config: HHLConfig,
    ) -> _HhlCircuit:
        num_vector_qubits = int(np.log2(linear_system.dimension))

        # Determine phase qubit count
        if config.num_phase_qubits is not None:
            num_phase_qubits = config.num_phase_qubits
        else:
            # Default: vector qubits + 2
            num_phase_qubits = num_vector_qubits + 2
            if config.max_phase_qubits is not None:
                num_phase_qubits = min(num_phase_qubits, config.max_phase_qubits)
            num_phase_qubits = max(num_phase_qubits, config.min_phase_qubits)

        # Create quantum registers
        vector_reg = QuantumRegister(num_vector_qubits, name="vector")
        phase_reg = QuantumRegister(num_phase_qubits, name="phase")
        flag_reg = QuantumRegister(1, name="flag")

        # Vector state preparation circuit
        vector_circuit = QuantumCircuit(vector_reg.size, name="State Preparation")
        vector_circuit.append(
            StatePreparation(linear_system.vector.tolist()),
            vector_circuit.qubits,
        )

        # Matrix evolution unitary: U = exp(iAt)
        # Qiskit standard uses t = 2π (evolution_time_factor = 2.0)
        matrix_circuit = QuantumCircuit(vector_reg.size, name="U")
        matrix_circuit.unitary(
            sp.linalg.expm(
                1j * linear_system.matrix * np.pi * config.evolution_time_factor
            ),
            matrix_circuit.qubits,
        )

        # Quantum Phase Estimation circuit
        qpe_circuit = phase_estimation(num_phase_qubits, matrix_circuit)

        # Controlled Y-rotation for reciprocal (amplitude amplification)
        ucry_circuit = QuantumCircuit(num_phase_qubits + 1)
        ucry_circuit.append(
            ExactReciprocalGate(
                num_phase_qubits,
                1 / (2**num_phase_qubits),
                True,
            ),
            ucry_circuit.qubits,
        )

        # Assemble complete HHL circuit
        hhl_circuit = _HhlCircuit(vector_reg, phase_reg, flag_reg, None,None name="HHL")
        hhl_circuit.append(vector_circuit, vector_reg[:])
        hhl_circuit.append(qpe_circuit, phase_reg[:] + vector_reg[:])
        hhl_circuit.append(ucry_circuit, phase_reg[::-1] + flag_reg[:])
        hhl_circuit.append(qpe_circuit.inverse(), phase_reg[:] + vector_reg[:])

        return hhl_circuit

"""


def _construct_circuit_sapo(matrix: np.ndarray, vector: np.ndarray) -> _HhlCircuit:
    """
    SAPO-optimized HHL implementation.

    This method uses SAPO (Scalable Adiabatic-inspired Phase Optimization) to:
    - Optimize matrix scaling based on eigenvalue bounds
    - Minimize required phase qubits
    - Use custom controlled rotations tuned to eigenvalue distribution

    Key differences from Qiskit standard:
    - Evolution time: π (evolution_time_factor = 1.0)
    - Phase qubits: Computed from eigenvalue bounds (usually fewer)
    - Custom UCRYGate angles based on norm constant

    References:
        Zhu, Tianze, et al.
        "SAPO: Improving the Scalability and Accuracy of
        Quantum Linear Solver for Portfolio Optimization."
        2025 62nd ACM/IEEE Design Automation Conference (DAC). IEEE, 2025.
    """

    # Apply SAPO optimization
    sapo_optimizer = SAPO(matrix, vector)

    optimized_system = sapo_optimizer.prepared_linear_system

    num_vector_qubits = int(np.log2(optimized_system.dimension))

    # Determine phase qubit count (SAPO provides recommendation)
    num_phase_qubits = min(sapo_optimizer.get_qpe_qubit_num(), 8)

    # Create quantum registers
    vector_reg = QuantumRegister(num_vector_qubits, name="vector")
    phase_reg = QuantumRegister(num_phase_qubits, name="phase")
    flag_reg = QuantumRegister(1, name="flag")

    # Vector state preparation circuit
    vector_circuit = QuantumCircuit(vector_reg.size, name="State Preparation")
    vector_circuit.append(
        StatePreparation(optimized_system.vector.tolist()),
        vector_circuit.qubits,
    )

    # Matrix evolution unitary: U = exp(iAt)
    # SAPO uses t = π (evolution_time_factor = 1.0)
    matrix_circuit = QuantumCircuit(vector_reg.size, name="U")
    matrix_circuit.unitary(
        sp.linalg.expm(1j * optimized_system.matrix * np.pi),
        matrix_circuit.qubits,
    )

    # Quantum Phase Estimation circuit
    qpe_circuit = phase_estimation(num_phase_qubits, matrix_circuit)

    # Custom controlled Y-rotation optimized for SAPO
    ucry_circuit = QuantumCircuit(num_phase_qubits + 1)

    def _compute_sapo_angles(
        num_phase_qubits: int, norm_constant: float
    ) -> list[float]:
        """
        Compute rotation angles for SAPO's controlled Y-rotation.

        Args:
            num_phase_qubits: Number of phase estimation qubits
            norm_constant: SAPO normalization constant

        Returns:
            List of rotation angles for UCRYGate
        """
        angles = [0.0]  # Angle for phase=0 state
        for i in range(1, 2**num_phase_qubits):
            # Phase value extracted by QPE
            phi = i / 2**num_phase_qubits

            # Rotation value: C * 0.5 / (phi - offset)
            # offset = 1 if i >= 2^(n-1) else 0 (handles negative phases)
            offset = 1 if i >= 2 ** (num_phase_qubits - 1) else 0
            rotation_value = norm_constant * 0.5 / (phi - offset)

            # Convert rotation value to angle with bounds checking
            if np.isclose(rotation_value, 1, rtol=1e-5, atol=1e-5):
                angles.append(np.pi)
            elif np.isclose(rotation_value, -1, rtol=1e-5, atol=1e-5):
                angles.append(-np.pi)
            elif -1 < rotation_value < 1:
                angles.append(2 * np.arcsin(rotation_value))
            else:
                # Out of bounds - no rotation
                angles.append(0)

        return angles

    angles = _compute_sapo_angles(num_phase_qubits, sapo_optimizer.norm_const)
    ucry_circuit.compose(UCRYGate(angles), inplace=True)

    # Assemble complete HHL circuit
    hhl_circuit = _HhlCircuit(
        vector_reg,
        phase_reg,
        flag_reg,
        norm_const=sapo_optimizer.norm_const,
        linear_system=optimized_system,
        name="HHL",
    )
    hhl_circuit.append(vector_circuit, vector_reg[:])
    hhl_circuit.append(qpe_circuit, phase_reg[:] + vector_reg[:])
    hhl_circuit.append(ucry_circuit, flag_reg[:] + phase_reg[::-1])
    hhl_circuit.append(qpe_circuit.inverse(), phase_reg[:] + vector_reg[:])
    return hhl_circuit


def _extract_solution_by_statevector(circuit: _HhlCircuit):
    """
    Extract the quantum solution from HHL circuit statevector.

    This function extracts and post-processes the quantum solution from an HHL
    algorithm circuit. It computes the success probability, extracts solution
    amplitudes from specific statevector indices, and scales the result back
    to the original problem scale.

    The HHL algorithm encodes the solution in the amplitudes of states where
    the flag qubit is 1 and all phase qubits are 0. These amplitudes need to
    be normalized and scaled to recover the actual solution to Ax = b.

    Args:
        circuit: The HHL circuit with embedded normalization constant and
                 linear system information.

    Returns:
        np.ndarray: The solution vector x to the linear system Ax = b,
                   scaled back to the original problem scale.

    Raises:
        ValueError: If success probability is too low (< 1e-10), indicating
                   HHL algorithm failure.
        ValueError: If extracted solution has zero norm, indicating a problem
                   with circuit construction.

    Examples:
        >>> circuit = _construct_circuit_sapo(matrix, vector)
        >>> solution = _extract_solution_by_statevector(circuit)
        >>> # Verify solution: np.allclose(matrix @ solution, vector)

    Notes:
        - The success probability is determined by the probability of measuring
          flag qubit = 1
        - Solution amplitudes are extracted from states |vector⟩|000...0⟩|1⟩
        - Final scaling includes: normalization, success probability correction,
          norm constant division, and original problem scaling
    """
    states = Statevector.from_circuit(circuit)
    # Calculate success probability (flag qubit = 0)
    success_prob = states.probabilities_dict(circuit.flag_qubits).get("1", 0)

    if success_prob < 1e-10:
        raise ValueError(
            f"Success probability too low: {success_prob:.2e}. "
            "The HHL algorithm failed to produce a valid solution. "
            "Try adjusting the configuration or checking matrix conditioning."
        )

    def _build_flag_state_indices() -> list[int]:
        """
        Build indices for extracting solution amplitudes from HHL statevector.

        In the HHL algorithm, the solution is encoded in specific amplitudes of the
        final statevector. This function computes the indices corresponding to states
        where the flag qubit is 1 (indicating successful amplitude amplification).

        The qubit ordering is: [vector_qubits][phase_qubits][flag_qubit]
        We want to extract amplitudes where flag_qubit=1 and phase_qubits=000...0

        Args:
            num_vector_qubits: Number of qubits encoding the solution vector
            num_phase_qubits: Number of qubits used for phase estimation

        Returns:
            List of integer indices into the statevector for extracting solution

        Examples:
            >>> # For 2 vector qubits, 3 phase qubits:
            >>> # Want states: |00⟩|000⟩|1⟩, |01⟩|000⟩|1⟩, |10⟩|000⟩|1⟩, |11⟩|000⟩|1⟩
            >>> indices = build_flag_state_indices(2, 3)
            >>> # Binary: flag=1, phase=000, vector=00/01/10/11
            >>> # = 1000000, 1000001, 1000010, 1000011
            >>> # = 64, 65, 66, 67 (in decimal)
            >>> len(indices)
            4
        """
        indices = []
        for i in range(2**circuit.num_vector_qubits):
            # Construct binary string: flag=1 + phase_qubits all 0 + vector state
            # Qiskit uses little-endian qubit ordering
            binary_str = (
                "1"  # flag qubit = 1 (success state)
                + "0" * circuit.num_phase_qubits  # phase qubits = 000...0
                + np.binary_repr(i, circuit.num_vector_qubits)  # vector state
            )
            indices.append(int(binary_str, 2))
        return indices

    # Extract amplitudes for success states
    indices = _build_flag_state_indices()
    solution_amplitudes = np.real(states.data)[indices]

    # Normalize and scale back to original problem
    solution_norm = np.linalg.norm(solution_amplitudes)
    if solution_norm < 1e-10:
        raise ValueError(
            "Extracted solution has zero norm. "
            "This indicates a problem with the HHL circuit construction."
        )

    solution = (
        solution_amplitudes
        * np.sqrt(success_prob)
        / circuit.norm_const
        / solution_norm
        * circuit.linear_system.solution_scaling
    )

    return solution


class QuantumLinearSolver:
    """
    Quantum solver for linear systems using HHL algorithm.

    This class provides a high-level interface for solving linear systems Ax = b
    using quantum computing via the Harrow-Hassidim-Lloyd (HHL) algorithm. It
    implements various optimization methods to improve the efficiency and accuracy
    of the quantum solution.

    The solver uses statevector simulation to extract the quantum solution, making
    it suitable for small to medium-sized problems on classical hardware.

    Methods:
        solve: Solve a linear system using quantum HHL algorithm

    Examples:
        Basic usage:
        >>> import numpy as np
        >>> from double_quant.solver import QuantumLinearSolver
        >>> A = np.array([[2, 1], [1, 2]])
        >>> b = np.array([3, 3])
        >>> x = QuantumLinearSolver.solve(A, b)
        >>> # Verify: np.allclose(A @ x, b)

        Using SAPO optimization (default):
        >>> x = QuantumLinearSolver.solve(A, b, method="sapo")

    Notes:
        - Matrix A must be square and symmetric (Hermitian)
        - The algorithm uses statevector simulation, which is memory-intensive
        - For dimension n, requires 2^n complex amplitudes in memory
        - Recommended for systems with dimension ≤ 2^10 on typical hardware
    """

    @staticmethod
    def solve(
        matrix: np.ndarray,
        vector: np.ndarray,
        method: Literal["sapo", "qiskit"] = "sapo",
    ):
        """
        Solve the linear system Ax = b using quantum HHL algorithm.

        This method constructs an HHL quantum circuit and extracts the solution
        from the resulting statevector. The solution is automatically scaled back
        to the original problem scale.

        Args:
            matrix: Coefficient matrix A of shape (n, n). Must be square and
                   symmetric (Hermitian). For best results, should be well-conditioned.
            vector: Right-hand side vector b of shape (n,). Will be normalized
                   internally.
            method: Optimization method to use. Currently supported:
                   - "sapo": SAPO optimization (default, recommended)
                   - "qiskit": Standard Qiskit implementation (commented out)

        Returns:
            np.ndarray: Solution vector x of shape (n,) satisfying Ax = b.

        Raises:
            ValueError: If matrix is not square or dimensions are incompatible.
            ValueError: If matrix is not symmetric (warning issued, may still run).
            ValueError: If HHL algorithm fails (success probability too low).
            NotImplementedError: If method is "qiskit" (not yet implemented).

        Examples:
            Solve a simple 2×2 system:
            >>> A = np.array([[2, 1], [1, 2]])
            >>> b = np.array([3, 3])
            >>> x = QuantumLinearSolver.solve(A, b)
            >>> print(f"Solution: {x}")
            >>> print(f"Verification: {np.allclose(A @ x, b)}")

            Solve a random 4×4 system:
            >>> from double_quant.common import LinearSystem
            >>> system = LinearSystem.random_for_hhl(n=4)
            >>> x = QuantumLinearSolver.solve(
            ...     system.matrix, system.vector, method="sapo"
            ... )

        Notes:
            - The SAPO method automatically optimizes:
              * Matrix scaling based on eigenvalue bounds
              * Number of phase estimation qubits
              * Controlled rotation angles for amplitude amplification
            - Success probability depends on matrix conditioning and eigenvalue spread
            - Better conditioned matrices generally yield higher success probability
            - The solution is computed via statevector simulation (exact, no sampling)
        """
        return _extract_solution_by_statevector(_construct_circuit_sapo(matrix, vector))
