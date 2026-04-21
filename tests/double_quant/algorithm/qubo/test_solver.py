from itertools import product

import numpy as np
import pytest
from qiskit_algorithms.optimizers import COBYLA

from double_quant.algorithm.qubo import (
    NumPyMinimumEigensolverSolver,
    QAOASolver,
    SamplingVQESolver,
    bits_to_spins,
    qubo_to_ising,
)
from double_quant.common import QUBOProblem


class TestQuboTranslation:
    def test_qubo_to_ising_preserves_objective(self):
        problem = QUBOProblem(
            np.array(
                [
                    [1.0, 2.0, -1.0],
                    [0.5, -3.0, 4.0],
                    [2.0, 1.5, 0.25],
                ]
            ),
            constant=1.25,
            variable_names=["a", "b", "c"],
        )
        ising = qubo_to_ising(problem)

        assert ising.variable_names == problem.variable_names

        for bits in product([0, 1], repeat=problem.num_variables):
            bit_array = np.array(bits, dtype=int)
            spins = bits_to_spins(bit_array)
            assert problem.evaluate(bit_array) == pytest.approx(ising.evaluate(spins))


class TestExactBaseline:
    def test_numpy_minimum_eigensolver_matches_bruteforce_optimum(self):
        problem = QUBOProblem(np.array([[-1.0, 2.0], [0.0, -0.5]]))

        result = NumPyMinimumEigensolverSolver().solve(problem)

        brute_force = {
            bits: problem.evaluate(np.array(bits, dtype=int))
            for bits in product([0, 1], repeat=problem.num_variables)
        }
        expected_bits, expected_value = min(brute_force.items(), key=lambda item: item[1])

        assert result.best_bitstring.tolist() == list(expected_bits)
        assert result.best_objective == pytest.approx(expected_value)
        assert result.best_energy == pytest.approx(expected_value)
        assert result.best_probability == pytest.approx(1.0)
        assert result.probabilities is not None
        assert result.probabilities["10"] == pytest.approx(1.0)
        for bitstring, probability in result.probabilities.items():
            if bitstring != "10":
                assert probability < 1.0e-20


class TestVariationalSolvers:
    def test_qaoa_solver_finds_single_variable_qubo_optimum(self):
        problem = QUBOProblem(np.array([[-1.0]]))

        result = QAOASolver(optimizer=COBYLA(maxiter=8), reps=1, seed=7).solve(problem)

        assert result.best_bitstring.tolist() == [1]
        assert result.best_objective == pytest.approx(-1.0)
        assert result.best_energy == pytest.approx(-1.0)
        assert result.parameter_values is not None
        assert result.metadata is not None

    def test_qaoa_solver_finds_two_variable_qubo_optimum(self):
        problem = QUBOProblem(np.array([[-1.0, 0.0], [0.0, -1.5]]))

        result = QAOASolver(optimizer=COBYLA(maxiter=20), reps=1, seed=7).solve(problem)

        assert result.best_bitstring.tolist() == [1, 1]
        assert result.best_objective == pytest.approx(-2.5)
        assert result.best_energy == pytest.approx(-2.5)
        assert result.parameter_values is not None
        assert result.metadata is not None

    def test_sampling_vqe_solver_finds_single_variable_qubo_optimum(self):
        problem = QUBOProblem(np.array([[-1.0]]))

        result = SamplingVQESolver(optimizer=COBYLA(maxiter=8), seed=7).solve(problem)

        assert result.best_bitstring.tolist() == [1]
        assert result.best_objective == pytest.approx(-1.0)
        assert result.best_energy == pytest.approx(-1.0)
        assert result.parameter_values is not None
        assert result.metadata is not None

    def test_sampling_vqe_solver_finds_two_variable_qubo_optimum(self):
        problem = QUBOProblem(np.array([[-1.0, 0.0], [0.0, -1.5]]))

        result = SamplingVQESolver(optimizer=COBYLA(maxiter=20), seed=7).solve(problem)

        assert result.best_bitstring.tolist() == [1, 1]
        assert result.best_objective == pytest.approx(-2.5)
        assert result.best_energy == pytest.approx(-2.5)
        assert result.parameter_values is not None
        assert result.metadata is not None
