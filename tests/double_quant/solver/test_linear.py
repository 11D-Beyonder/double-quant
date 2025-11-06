import numpy as np
import pytest
from double_quant.solver.linear import (
    NumPyLinearSolver,
    SciPyLinearSolver,
)


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
