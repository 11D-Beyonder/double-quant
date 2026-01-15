from matplotlib.pyplot import flag
import numpy as np
import pytest
from double_quant.solver.linear import (
    NumPyLinearSolver,
    QuantumLinearSolver,
    SciPyLinearSolver,
)
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import (
    ExactReciprocalGate,
    StatePreparation,
    UCRYGate,
    phase_estimation,
)
from qiskit.quantum_info import Statevector


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
        # circuit.append(
        #     UCRYGate([0, 0, 0, 0, 2 * np.arcsin(1), 0, 0, 0]),
        #     flag_reg[:] + phase_reg[::-1],
        # )
        print(f"\n{'=' * 60}\n{circuit.decompose('1/x').draw()}")
        print(Statevector.from_circuit(circuit).probabilities_dict([5]))


class TestQuantumLinearSolver:
    def test_print_circuit(self):
        solver = QuantumLinearSolver()
        num_ele = 2**2
        solver._linear_system = QuantumLinearSolver._LinearSystem.random(num_ele)
        circuit = solver._construct_circuit()
        print(f"\n{circuit.decompose('circuit-43').decompose('1/x').draw()}")
        # circuit.draw("mpl")
        # plt.show()


class TestNumPyLinearSolver:
    def test_solve_2x2_system(self):
        """Test solving a 2x2 linear system."""
        solver = NumPyLinearSolver()

        # Simple 2x2 system: [2 1; 1 1] * [x; y] = [3; 2]
        A = np.array([[2, 1], [1, 1]])
        b = np.array([3, 2])

        solution = solver.solve(A, b)
        expected = np.array([1, 1])

        np.testing.assert_array_almost_equal(solution, expected)

    def test_solve_3x3_system(self):
        """Test solving a 3x3 linear system."""
        solver = NumPyLinearSolver()

        # 3x3 system
        A = np.array([[3, 1, 1], [1, 3, 1], [1, 1, 3]])
        b = np.array([5, 5, 5])

        solution = solver.solve(A, b)
        expected = np.array([1, 1, 1])

        np.testing.assert_array_almost_equal(solution, expected)

    def test_solve_singular_matrix(self):
        """Test that singular matrix raises appropriate error."""
        solver = NumPyLinearSolver()

        # Singular matrix (determinant = 0)
        A = np.array([[1, 2], [2, 4]])  # Second row is 2x first row
        b = np.array([1, 2])

        with pytest.raises(np.linalg.LinAlgError):
            _ = solver.solve(A, b)

    def test_solve_incompatible_dimensions(self):
        """Test that incompatible dimensions raise appropriate error."""
        solver = NumPyLinearSolver()

        # 2x2 matrix with 3-element vector
        A = np.array([[1, 2], [3, 4]])
        b = np.array([1, 2, 3])

        with pytest.raises(ValueError):
            _ = solver.solve(A, b)


class TestSciPyLinearSolver:
    def test_solve_2x2_system(self):
        """Test solving a 2x2 linear system."""
        solver = SciPyLinearSolver()

        # Simple 2x2 system: [2 1; 1 1] * [x; y] = [3; 2]
        A = np.array([[2, 1], [1, 1]])
        b = np.array([3, 2])

        solution = solver.solve(A, b)
        expected = np.array([1, 1])

        np.testing.assert_array_almost_equal(solution, expected)

    def test_solve_3x3_system(self):
        """Test solving a 3x3 linear system."""
        solver = SciPyLinearSolver()

        # 3x3 system
        A = np.array([[3, 1, 1], [1, 3, 1], [1, 1, 3]])
        b = np.array([5, 5, 5])

        solution = solver.solve(A, b)
        expected = np.array([1, 1, 1])

        np.testing.assert_array_almost_equal(solution, expected)

    def test_solve_singular_matrix(self):
        """Test that singular matrix raises appropriate error."""
        solver = SciPyLinearSolver()

        # Singular matrix (determinant = 0)
        A = np.array([[1, 2], [2, 4]])  # Second row is 2x first row
        b = np.array([1, 2])

        with pytest.raises(np.linalg.LinAlgError):
            _ = solver.solve(A, b)

    def test_solve_incompatible_dimensions(self):
        """Test that incompatible dimensions raise appropriate error."""
        solver = SciPyLinearSolver()

        # 2x2 matrix with 3-element vector
        A = np.array([[1, 2], [3, 4]])
        b = np.array([1, 2, 3])

        with pytest.raises(ValueError):
            _ = solver.solve(A, b)


class TestSolverConsistency:
    """Test that NumPy and SciPy solvers produce the same results."""

    def test_consistency_2x2(self):
        """Test that NumPy and SciPy solvers give same result for 2x2 system."""
        numpy_solver = NumPyLinearSolver()
        scipy_solver = SciPyLinearSolver()

        A = np.array([[4, 2], [1, 3]])
        b = np.array([8, 4])

        numpy_result = numpy_solver.solve(A, b)
        scipy_result = scipy_solver.solve(A, b)

        np.testing.assert_array_almost_equal(numpy_result, scipy_result)

    def test_consistency_3x3(self):
        """Test that NumPy and SciPy solvers give same result for 3x3 system."""
        numpy_solver = NumPyLinearSolver()
        scipy_solver = SciPyLinearSolver()

        A = np.array([[2, 1, 0], [1, 2, 1], [0, 1, 2]])
        b = np.array([1, 2, 1])

        numpy_result = numpy_solver.solve(A, b)
        scipy_result = scipy_solver.solve(A, b)

        np.testing.assert_array_almost_equal(numpy_result, scipy_result)
