from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

import numpy as np
from qiskit import QuantumCircuit
from numpy.typing import NDArray

from double_quant.algorithm.rasengan.linear_system import find_transition_basis
from double_quant.algorithm.rasengan.model import LinearConstraintBinaryProblem


def build_rasengan_circuit(
    problem: LinearConstraintBinaryProblem,
    *,
    layers: int = 1,
    transition_basis: NDArray[np.int_] | None = None,
    feasible_state: NDArray[np.int_] | list[int] | None = None,
    phase_scale: float = math.pi / 6,
    transition_phases: NDArray[np.float64] | Sequence[float] | None = None,
    measure: bool = True,
) -> QuantumCircuit:
    """Build the Rasengan transition-Hamiltonian circuit for a binary model."""
    if layers <= 0:
        raise ValueError("layers must be positive")
    basis = (
        find_transition_basis(problem.constraints)
        if transition_basis is None
        else np.asarray(transition_basis, dtype=int)
    )
    if basis.ndim != 2 or basis.shape[1] != problem.num_variables:
        raise ValueError("transition_basis must be a matrix with n columns")

    logical_qubits = problem.num_variables
    ancilla_qubits = _required_ancillas(basis)
    classical_bits = logical_qubits if measure else 0
    circuit = QuantumCircuit(
        logical_qubits + ancilla_qubits,
        classical_bits,
        name=f"rasengan_n{logical_qubits}_p{layers}",
    )
    ancillas = list(range(logical_qubits, logical_qubits + ancilla_qubits))

    initial_state = (
        np.asarray(feasible_state, dtype=int)
        if feasible_state is not None
        else problem.feasible_state()
    )
    if initial_state.shape != (problem.num_variables,):
        raise ValueError("feasible_state length must match variable count")
    if not problem.is_feasible(initial_state):
        raise ValueError("feasible_state must satisfy the problem constraints")
    for index, bit in enumerate(initial_state):
        if int(bit) == 1:
            circuit.x(index)

    phase_matrix = _transition_phase_matrix(
        layers=layers,
        transition_count=len(basis),
        transition_phases=transition_phases,
        phase_scale=phase_scale,
    )
    if len(basis) > 0:
        for layer in range(layers):
            for transition_index, transition in enumerate(basis):
                phase = float(phase_matrix[layer, transition_index])
                _append_driver_component(circuit, transition, ancillas, phase)

    if measure:
        circuit.measure(range(logical_qubits), range(logical_qubits)[::-1])

    circuit.metadata = {
        "algorithm": "rasengan",
        "source": "double_quant.algorithm.rasengan",
        "external_rasengan_dependency": False,
        "num_variables": logical_qubits,
        "num_constraints": problem.num_constraints,
        "layers": layers,
        "transition_count": int(len(basis)),
        "transition_parameter_count": int(phase_matrix.size),
        "ancilla_qubits": ancilla_qubits,
    }
    return circuit


def _transition_phase_matrix(
    *,
    layers: int,
    transition_count: int,
    transition_phases: NDArray[np.float64] | Sequence[float] | None,
    phase_scale: float,
) -> NDArray[np.float64]:
    if transition_count == 0:
        return np.zeros((layers, 0), dtype=float)
    if transition_phases is None:
        return np.full((layers, transition_count), phase_scale, dtype=float)

    phases = np.asarray(transition_phases, dtype=float)
    if phases.ndim == 0:
        raise ValueError("transition_phases must contain one value per transition")
    if phases.ndim == 1:
        if phases.size == transition_count:
            return np.tile(phases.reshape(1, transition_count), (layers, 1))
        if phases.size == layers * transition_count:
            return phases.reshape(layers, transition_count)
    if phases.ndim == 2 and phases.shape == (layers, transition_count):
        return phases
    raise ValueError(
        "transition_phases must have length transition_count, "
        "layers * transition_count, or shape (layers, transition_count)"
    )


def _required_ancillas(transition_basis: NDArray[np.int_]) -> int:
    if len(transition_basis) == 0:
        return 1
    max_support = max(np.count_nonzero(row) for row in transition_basis)
    return 2 if max_support > 2 else 1


def _append_driver_component(
    circuit: QuantumCircuit,
    transition: NDArray[np.int_],
    ancillas: Sequence[int],
    phase: float,
) -> None:
    support = np.nonzero(transition)[0].tolist()
    if not support:
        return
    bit_string = [0 if transition[index] == -1 else 1 for index in support]
    _apply_convert(circuit, support, bit_string)
    _append_phase_gate(circuit, support, ancillas, -phase)
    circuit.x(support[-1])
    _append_phase_gate(circuit, support, ancillas, phase)
    circuit.x(support[-1])
    _apply_reverse(circuit, support, bit_string)


def _apply_convert(
    circuit: QuantumCircuit,
    qubits: Sequence[int],
    bit_string: Sequence[int],
) -> None:
    for index in range(len(bit_string) - 1):
        circuit.cx(qubits[index + 1], qubits[index])
        if bit_string[index] == bit_string[index + 1]:
            circuit.x(qubits[index])
    circuit.h(qubits[-1])
    circuit.x(qubits[-1])


def _apply_reverse(
    circuit: QuantumCircuit,
    qubits: Sequence[int],
    bit_string: Sequence[int],
) -> None:
    circuit.x(qubits[-1])
    circuit.h(qubits[-1])
    for index in range(len(bit_string) - 2, -1, -1):
        if bit_string[index] == bit_string[index + 1]:
            circuit.x(qubits[index])
        circuit.cx(qubits[index + 1], qubits[index])


def _append_phase_gate(
    circuit: QuantumCircuit,
    qubits: Sequence[int],
    ancillas: Sequence[int],
    phase: float,
) -> None:
    if len(qubits) == 1:
        circuit.p(phase, qubits[0])
    elif len(qubits) == 2:
        circuit.cp(phase, qubits[0], qubits[1])
    else:
        _append_decomposed_multi_phase(circuit, list(qubits), list(ancillas), phase)


def _append_decomposed_multi_phase(
    circuit: QuantumCircuit,
    qubits: list[int],
    ancillas: list[int],
    phase: float,
) -> None:
    if len(ancillas) < 2:
        circuit.mcp(phase, qubits[:-1], qubits[-1])
        return
    split = len(qubits) // 2
    left_controls = qubits[:split]
    right_controls = qubits[split:]
    target = ancillas[0]
    rest = ancillas[1:]
    circuit.rz(-phase / 2, target)
    _append_mcx(circuit, left_controls, target, rest)
    circuit.rz(phase / 2, target)
    _append_mcx(circuit, right_controls, target, rest)
    circuit.rz(-phase / 2, target)
    _append_mcx(circuit, left_controls, target, rest)
    circuit.rz(phase / 2, target)
    _append_mcx(circuit, right_controls, target, rest)


def _append_mcx(
    circuit: QuantumCircuit,
    controls: Iterable[int],
    target: int,
    ancillas: Sequence[int],
) -> None:
    controls_list = list(controls)
    if len(controls_list) == 0:
        circuit.x(target)
    elif len(controls_list) == 1:
        circuit.cx(controls_list[0], target)
    elif len(controls_list) == 2:
        circuit.ccx(controls_list[0], controls_list[1], target)
    else:
        circuit.mcx(controls_list, target, ancillas)
