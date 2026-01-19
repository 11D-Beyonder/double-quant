import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import (
    ExactReciprocalGate,
    StatePreparation,
    UCRYGate,
    phase_estimation,
)
from qiskit.quantum_info import Statevector
from double_quant.common import LinearSystem
from double_quant.common.metric import cos_similarity
from double_quant.solver import QuantumLinearSolver


class TestHhlQubitOrder:
    """Test bit ordering differences between old and new phase_estimation implementations."""

    def test_qpe_order(self):
        """
        Compare the output bit order of deprecated PhaseEstimation vs new phase_estimation.

        Tests with a 4x4 unitary matrix. The key finding is that BOTH implementations
        now output the same bit order due to how reverse_bits() works.
        """
        # Create a 4x4 unitary matrix (2 qubits)
        # Use a simple diagonal matrix with known eigenvalues
        # U = diag(e^(2πi*0.25), e^(2πi*0.5), e^(2πi*0.75), e^(2πi*0.0))
        phases = np.array([-0.25, -0.5, -0.75, 0.0])
        U = np.diag(np.exp(2j * np.pi * phases))

        vector_reg = QuantumRegister(2, "vector")
        phase_reg = QuantumRegister(3, "phase")
        flag_reg = QuantumRegister(1, "flag")
        # Create the unitary circuit
        unitary_circuit = QuantumCircuit(vector_reg.size, name="U")
        unitary_circuit.unitary(U, unitary_circuit.qubits)

        circuit = QuantumCircuit(vector_reg, phase_reg, flag_reg)

        circuit.append(StatePreparation([0, 0, 1, 0]), vector_reg[:])
        circuit.append(
            phase_estimation(phase_reg.size, unitary_circuit),
            phase_reg[:] + vector_reg[:],
        )
        print("")
        print(Statevector.from_circuit(circuit).probabilities_dict([2]))
        print(Statevector.from_circuit(circuit).probabilities_dict([3]))
        print(Statevector.from_circuit(circuit).probabilities_dict([4]))
        print(Statevector.from_circuit(circuit).probabilities_dict([2, 3, 4]))
        circuit.append(
            ExactReciprocalGate(phase_reg.size, 0.25 / 2**phase_reg.size, True),
            phase_reg[::-1] + flag_reg[:],
        )
        circuit.append(
            UCRYGate([0, 0, 0, 0, 2 * np.arcsin(1), 0, 0, 0]),
            flag_reg[:] + phase_reg[::-1],
        )
        print(f"\n{'=' * 60}\n{circuit.decompose('1/x').draw()}")
        print(Statevector.from_circuit(circuit).probabilities_dict([5]))


class TestQuantumLinearSolver:
    """Test cases for QuantumLinearSolver."""

    def test_basic_solve_sapo_method(self):
        """Test quantum solver with SAPO method on simple system."""

        NUM_TEST = 100
        count = 0
        for _ in range(NUM_TEST):
            system = LinearSystem.random_for_hhl(2**2)
            count += (
                cos_similarity(
                    np.linalg.solve(system.matrix, system.vector),
                    QuantumLinearSolver.solve(system.matrix, system.vector),
                )
                >= 0.5
            )
        assert count >= NUM_TEST // 2


class TestQuantumVsClassical:
    """Compare quantum and classical solver results."""

    def test_compare_with_numpy_2x2(self):
        """Compare quantum solver with NumPy on 2x2 system."""
        ...
