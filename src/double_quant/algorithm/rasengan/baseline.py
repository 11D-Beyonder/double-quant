from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from qiskit import QuantumCircuit

from double_quant.algorithm.rasengan.model import LinearConstraintBinaryProblem


def build_penalty_qaoa_circuit(
    problem: LinearConstraintBinaryProblem,
    *,
    layers: int = 1,
    gamma: float | Sequence[float] = math.pi / 7,
    beta: float | Sequence[float] = math.pi / 5,
    measure: bool = True,
) -> QuantumCircuit:
    """Build a deterministic Penalty-QAOA circuit for the same constrained model."""
    if layers <= 0:
        raise ValueError("layers must be positive")
    num_qubits = problem.num_variables
    circuit = QuantumCircuit(
        num_qubits,
        num_qubits if measure else 0,
        name=f"penalty_qaoa_n{num_qubits}_p{layers}",
    )
    circuit.h(range(num_qubits))
    gamma_values = _layer_parameters(gamma, layers)
    beta_values = _layer_parameters(beta, layers)

    for layer in range(layers):
        scale = float(gamma_values[layer])
        _append_objective_phase(circuit, problem, scale)
        _append_penalty_phase(circuit, problem, scale)
        for qubit in range(num_qubits):
            circuit.rx(2 * float(beta_values[layer]), qubit)

    if measure:
        circuit.measure(range(num_qubits), range(num_qubits)[::-1])

    circuit.metadata = {
        "algorithm": "penalty_qaoa",
        "num_variables": num_qubits,
        "num_constraints": problem.num_constraints,
        "layers": layers,
        "parameter_count": 2 * layers,
        "penalty": problem.penalty,
    }
    return circuit


def _layer_parameters(
    value: float | Sequence[float],
    layers: int,
) -> np.ndarray:
    if isinstance(value, (int, float)):
        return np.asarray([float(value) / (layer + 1) for layer in range(layers)])
    values = np.asarray(value, dtype=float)
    if values.shape != (layers,):
        raise ValueError("QAOA parameter sequences must have one value per layer")
    return values


def _append_objective_phase(
    circuit: QuantumCircuit,
    problem: LinearConstraintBinaryProblem,
    gamma: float,
) -> None:
    direction = problem.objective_direction
    for qubit, coefficient in enumerate(problem.linear):
        if coefficient != 0:
            circuit.rz(-direction * gamma * float(coefficient), qubit)
    if problem.quadratic is None:
        return
    quadratic = np.asarray(problem.quadratic, dtype=float)
    for row in range(problem.num_variables):
        for column in range(row + 1, problem.num_variables):
            coefficient = float(quadratic[row, column] + quadratic[column, row])
            if coefficient != 0:
                _append_zz_phase(circuit, row, column, direction * gamma * coefficient / 2)


def _append_penalty_phase(
    circuit: QuantumCircuit,
    problem: LinearConstraintBinaryProblem,
    gamma: float,
) -> None:
    for row, rhs in zip(problem.constraints, problem.rhs):
        row = np.asarray(row, dtype=float)
        linear_coefficients = problem.penalty * (row**2 - 2 * rhs * row)
        for qubit, coefficient in enumerate(linear_coefficients):
            if coefficient != 0:
                circuit.rz(-gamma * float(coefficient), qubit)
        for left in range(problem.num_variables):
            for right in range(left + 1, problem.num_variables):
                coefficient = 2 * problem.penalty * row[left] * row[right]
                if coefficient != 0:
                    _append_zz_phase(circuit, left, right, gamma * float(coefficient) / 2)


def _append_zz_phase(
    circuit: QuantumCircuit,
    left: int,
    right: int,
    angle: float,
) -> None:
    circuit.cx(left, right)
    circuit.rz(angle, right)
    circuit.cx(left, right)
