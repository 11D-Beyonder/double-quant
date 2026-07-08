import numpy as np
import pytest

from simulator.metrics import (
    bitstring_objective_accuracy,
    distribution_total_variation,
    fidelity,
    solution_success_probability,
)
from simulator.models import SimulationBackend
from simulator.analysis import verify_capacity


def test_fidelity_for_equal_states_is_one():
    state = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)

    assert fidelity(state, state) == pytest.approx(1.0)


def test_distribution_total_variation_normalizes_counts():
    assert distribution_total_variation(
        {"0": 8, "1": 2}, {"0": 5, "1": 5}
    ) == pytest.approx(0.3)


def test_solution_success_probability_accepts_counts():
    assert solution_success_probability({"00": 2, "11": 6}, {"11"}) == pytest.approx(0.75)


def test_bitstring_objective_accuracy_is_distribution_weighted():
    objective = lambda bitstring: int(bitstring, 2)

    assert bitstring_objective_accuracy(
        {"00": 0.25, "10": 0.75}, objective, 2.0
    ) == pytest.approx(0.5)


def test_verify_capacity_accepts_statevector_cpu_20_qubits():
    report = verify_capacity(20, SimulationBackend.STATEVECTOR_CPU)

    assert report.supported is True
