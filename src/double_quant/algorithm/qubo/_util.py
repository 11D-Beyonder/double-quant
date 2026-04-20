from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from qiskit.primitives import StatevectorSampler
from qiskit_algorithms.optimizers import COBYLA

from double_quant.algorithm.qubo.result import QUBOSolverResult
from double_quant.algorithm.qubo.translate import (
    array_to_bitstring,
    bits_to_spins,
    ising_to_pauli_operator,
    qiskit_bitstring_to_array,
    qubo_to_ising,
)
from double_quant.common import IsingProblem, QUBOProblem


def default_sampler(seed: int | None) -> StatevectorSampler:
    return StatevectorSampler(seed=seed)


def default_optimizer() -> COBYLA:
    return COBYLA(maxiter=100)


def ensure_ising_problem(problem: QUBOProblem | IsingProblem) -> IsingProblem:
    if isinstance(problem, QUBOProblem):
        return qubo_to_ising(problem)
    return problem


def build_pauli_operator(problem: QUBOProblem | IsingProblem):
    return ising_to_pauli_operator(ensure_ising_problem(problem))


def build_result(
    source_problem: QUBOProblem | IsingProblem,
    ising_problem: IsingProblem,
    raw_result: Any,
) -> QUBOSolverResult:
    measurement = getattr(raw_result, "best_measurement", None)
    if not measurement:
        raise ValueError("Qiskit result does not include best_measurement")

    bits = qiskit_bitstring_to_array(str(measurement["bitstring"]))
    spins = bits_to_spins(bits)
    best_energy = ising_problem.evaluate(spins)
    if isinstance(source_problem, QUBOProblem):
        best_objective = source_problem.evaluate(bits)
    else:
        best_objective = best_energy

    probabilities = normalize_probabilities(getattr(raw_result, "eigenstate", None))
    parameter_values, parameter_names = extract_parameter_values(
        getattr(raw_result, "optimal_parameters", None)
    )
    metadata = {
        "raw_bitstring": str(measurement["bitstring"]),
        "project_bitstring": array_to_bitstring(bits),
        "best_measurement_value": float(np.real(measurement["value"])),
        "optimizer_time": getattr(raw_result, "optimizer_time", None),
        "cost_function_evals": getattr(raw_result, "cost_function_evals", None),
        "optimizer_evals": getattr(raw_result, "optimizer_evals", None),
        "parameter_names": parameter_names,
    }

    return QUBOSolverResult(
        best_bitstring=bits,
        best_objective=best_objective,
        best_energy=best_energy,
        best_probability=measurement.get("probability"),
        parameter_values=parameter_values,
        probabilities=probabilities,
        metadata=metadata,
    )


def normalize_probabilities(
    eigenstate: Mapping[str, float] | None,
) -> dict[str, float] | None:
    if eigenstate is None:
        return None
    normalized: dict[str, float] = {}
    for raw_bitstring, probability in eigenstate.items():
        bits = qiskit_bitstring_to_array(raw_bitstring)
        normalized[array_to_bitstring(bits)] = float(probability)
    return normalized


def extract_parameter_values(
    optimal_parameters: Mapping[Any, Any] | None,
) -> tuple[np.ndarray | None, list[str] | None]:
    if optimal_parameters is None:
        return None, None
    parameter_names = [str(parameter) for parameter in optimal_parameters]
    values = np.asarray(list(optimal_parameters.values()), dtype=float)
    return values, parameter_names
